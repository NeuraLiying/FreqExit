import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

# Global DWT / IDWT instances (avoid repeated creation)
_DWT = None
_IDWT = None
_DWT_DEVICE = None
_IDWT_DEVICE = None


def get_dwt(device):
    """Get or create DWT instance"""
    global _DWT, _DWT_DEVICE
    if _DWT is None or _DWT_DEVICE != device:
        _DWT = DWTForward(J=1, mode='reflect', wave='haar').to(device)
        _DWT_DEVICE = device
    return _DWT


def get_idwt(device):
    """Get or create IDWT instance"""
    global _IDWT, _IDWT_DEVICE
    if _IDWT is None or _IDWT_DEVICE != device:
        _IDWT = DWTInverse(mode='reflect', wave='haar').to(device)
        _IDWT_DEVICE = device
    return _IDWT


class SubbandAlignmentModule(nn.Module):
    """
    Subband Alignment Module: Predict high-frequency (LH, HL, HH) from low-frequency LL.
    Uses DWT/IDWT for reconstruction: r_aux = sam_module(r_t), loss = MSE(r_aux, r_t).
    """

    def __init__(self, channels: int = 32):
        super(SubbandAlignmentModule, self).__init__()
        # Predictor: LL (C channels) -> HF (3*C channels)
        self.predictor = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 3, kernel_size=1, bias=True)
        )
        # Initialize last layer to 0 for stable early training
        with torch.no_grad():
            self.predictor[-1].weight.zero_()
            self.predictor[-1].bias.zero_()

    def orthogonal_regularization(self):
        """Placeholder for compatibility, returns 0"""
        device = self.predictor[0].weight.device
        return torch.tensor(0.0, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> reconstructed: [B, C, H, W]"""
        B, C, H, W = x.shape
        device = x.device
        orig_dtype = x.dtype

        # Disable AMP for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            dwt = get_dwt(device)
            idwt = get_idwt(device)

            # Pad to even dimensions
            pad_right = W % 2
            pad_bottom = H % 2
            if pad_right or pad_bottom:
                mode = 'reflect' if min(H, W) > 1 else 'replicate'
                x_pad = F.pad(x, (0, pad_right, 0, pad_bottom), mode=mode)
            else:
                x_pad = x

            x_pad_fp32 = x_pad.float()

            # DWT decomposition
            LL, HF_list = dwt(x_pad_fp32)

            # Predict HF from LL
            pred_HF_flat = self.predictor(LL)  # [B, 3C, H/2, W/2]
            pred_HF = pred_HF_flat.view(B, C, 3, LL.shape[2], LL.shape[3])

            # Reconstruct using (LL, HF_pred)
            reconstructed = idwt((LL, [pred_HF]))

            # Crop to original size
            if pad_right or pad_bottom:
                reconstructed = reconstructed[:, :, :H, :W]

            return reconstructed.to(orig_dtype)


def frequency_guided_self_reconstruction_loss(
    student_token_maps,
    sample_masks,
    supervise_layers,
    sam_modules,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    all_steps=list(range(10)),
    orth_reg_weight: float = 0.1,
    diversity_weight: float = 0.01,
):
    """
    FGSR Loss: Frequency-Guided Self-Reconstruction Loss
    Uses SAM for LF->HF prediction with step-wise weighting and diversity regularization.
    """
    with torch.cuda.amp.autocast(enabled=False):
        total_fgsr_loss = 0.0
        total_reg_loss = 0.0
        num_valid_layers = 0

        # Get dtype from first student token map
        orig_dtype = None
        first_student = None
        if student_token_maps:
            first_student = next(iter(student_token_maps.values()))
            if first_student is not None and first_student.numel() > 0:
                orig_dtype = first_student.dtype

        # Calculate step indices
        step_indices = []
        cur_idx = 0
        for pn in patch_nums:
            step_indices.append((cur_idx, cur_idx + pn * pn))
            cur_idx += pn * pn

        device = first_student.device if first_student is not None else next(iter(sam_modules.values())).parameters().__next__().device

        # Loop over layers
        for layer_idx in supervise_layers:
            if layer_idx not in student_token_maps:
                continue

            mask = sample_masks[layer_idx]
            if mask is None or not mask.any():
                continue

            s_token_map = student_token_maps[layer_idx][mask]
            if s_token_map.numel() == 0:
                continue

            layer_fgsr_loss = 0.0
            layer_reg_loss = 0.0
            step_weights_sum = 0.0

            # Loop over steps
            for step_idx in all_steps:
                if step_idx >= len(patch_nums):
                    continue

                pn = patch_nums[step_idx]
                if pn < 8:  # Skip small scales
                    continue

                start_idx, end_idx = step_indices[step_idx]
                if end_idx > s_token_map.shape[1]:
                    continue

                slice_feat = s_token_map[:, start_idx:end_idx, :]
                if slice_feat.numel() == 0:
                    continue

                # Reshape: [B, pn*pn, C] -> [B', C, pn, pn]
                vae_channels = slice_feat.shape[-1]
                r_t = slice_feat.transpose(1, 2).reshape(-1, vae_channels, pn, pn)

                module_key = f"step_{step_idx}"
                if module_key not in sam_modules:
                    continue

                r_aux = sam_modules[module_key](r_t)
                mse_loss = F.mse_loss(r_aux, r_t)
                layer_reg_loss += sam_modules[module_key].orthogonal_regularization()

                # Step weighting: later steps have stronger constraints
                t = float(step_idx) / max(1, (len(patch_nums) - 1))
                step_weight = 0.7 + 0.6 * t
                layer_fgsr_loss += step_weight * mse_loss
                step_weights_sum += step_weight

            if step_weights_sum > 0:
                layer_fgsr_loss = layer_fgsr_loss / step_weights_sum
                total_fgsr_loss += layer_fgsr_loss
                total_reg_loss += layer_reg_loss / max(1, len(all_steps))
                num_valid_layers += 1

        if num_valid_layers > 0:
            avg_fgsr_loss = total_fgsr_loss / num_valid_layers
            avg_reg_loss = total_reg_loss / num_valid_layers
        else:
            avg_fgsr_loss = torch.tensor(0.0, device=device)
            avg_reg_loss = torch.tensor(0.0, device=device)

        # Diversity regularization: prevent all SAMs from learning the same weights
        diversity_loss = torch.tensor(0.0, device=device)
        step_keys = sorted(
            [k for k in sam_modules.keys() if k.startswith("step_")],
            key=lambda s: int(s.split("_")[1])
        )

        num_pairs = 0
        if len(step_keys) >= 2:
            for i in range(len(step_keys) - 1):
                m1 = sam_modules[step_keys[i]]
                m2 = sam_modules[step_keys[i + 1]]

                W1 = m1.predictor[-1].weight.view(-1)
                W2 = m2.predictor[-1].weight.view(-1)

                if W1.numel() == 0 or W2.numel() == 0:
                    continue

                W1_n = F.normalize(W1, dim=0)
                W2_n = F.normalize(W2, dim=0)
                cos_sim = (W1_n * W2_n).sum()
                diversity_loss += cos_sim * cos_sim
                num_pairs += 1

            if num_pairs > 0:
                diversity_loss = diversity_loss / num_pairs

        # Final loss
        final_fgsr_loss = avg_fgsr_loss + orth_reg_weight * avg_reg_loss + diversity_weight * diversity_loss

        if orig_dtype is not None:
            return final_fgsr_loss.to(orig_dtype)
        return final_fgsr_loss