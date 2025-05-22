import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_wavelets import DWTForward, DWTInverse



_DWT = None
_DWT_DEVICE = None 

def get_dwt(device):
    global _DWT, _DWT_DEVICE
    if _DWT is None or _DWT_DEVICE != device:
        _DWT = DWTForward(J=1, mode='reflect', wave='haar').to(device)
        _DWT_DEVICE = device
    return _DWT

def high_frequency_loss(
    teacher_token_map,
    student_token_maps,
    supervise_layers,
    sample_masks,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    hf_steps=(5, 6, 7, 8, 9),
    student_layer_weights=None,
):
  
    device = teacher_token_map.device
    vae_channels = teacher_token_map.shape[-1]
    orig_dtype = teacher_token_map.dtype  
    
  
    step_weight_power = 1.5
    
    
    with torch.cuda.amp.autocast(enabled=False):
        dwt = get_dwt(device)
        
        if student_layer_weights is None:
            student_layer_weights = {layer_idx: 1.0 for layer_idx in supervise_layers}
        
        layer_losses = {}
        valid_layer_weights = {}
        
        step_indices = []
        cur_idx = 0
        for pn in patch_nums:
            step_indices.append((cur_idx, cur_idx + pn*pn))
            cur_idx += pn*pn
        
        base_pn = patch_nums[hf_steps[-1]]
        
        for layer_idx in supervise_layers:
            if layer_idx not in student_token_maps:
                continue
                
            mask = sample_masks[layer_idx]
            if mask is None or not mask.any():
                continue
            
            s_token_map = student_token_maps[layer_idx]
            t_token_map = teacher_token_map.detach()
            
            s_token_map_valid = s_token_map[mask]
            t_token_map_valid = t_token_map[mask]
            
            layer_hf_loss = 0.0
            step_weights_sum = 0.0
            
            for step_idx in hf_steps:
                if step_idx >= len(patch_nums):
                    continue
                    
                pn = patch_nums[step_idx]
                start_idx, end_idx = step_indices[step_idx]
                
                s_slice = s_token_map_valid[:, start_idx:end_idx]
                t_slice = t_token_map_valid[:, start_idx:end_idx]
                s_reshape = s_slice.reshape(-1, vae_channels, pn, pn)
                t_reshape = t_slice.reshape(-1, vae_channels, pn, pn)
                
                s_reshape_fp32 = s_reshape.float()
                t_reshape_fp32 = t_reshape.float()
                
                s_coeffs = dwt(s_reshape_fp32)
                t_coeffs = dwt(t_reshape_fp32)
                
                s_H = s_coeffs[1][0]
                t_H = t_coeffs[1][0]
                s_high_freq = torch.cat([s_H[:, :, i] for i in range(3)], dim=1)
                t_high_freq = torch.cat([t_H[:, :, i] for i in range(3)], dim=1)
                
               
                hf_loss = F.mse_loss(s_high_freq, t_high_freq)
                
               
                step_weight = (pn / base_pn) ** step_weight_power
                
               
                layer_hf_loss += step_weight * hf_loss
                step_weights_sum += step_weight
            
           
            if step_weights_sum > 0:
                layer_hf_loss = layer_hf_loss / step_weights_sum
                layer_losses[layer_idx] = layer_hf_loss
                valid_layer_weights[layer_idx] = student_layer_weights[layer_idx]
        
     
        if layer_losses:
            total_layer_weight = sum(valid_layer_weights.values())
            total_hf_loss = sum(
                (valid_layer_weights[layer_idx] / total_layer_weight) * loss
                for layer_idx, loss in layer_losses.items()
            )
           
            return total_hf_loss.to(orig_dtype)
        else:
            return torch.tensor(0.0, device=device, dtype=orig_dtype)



