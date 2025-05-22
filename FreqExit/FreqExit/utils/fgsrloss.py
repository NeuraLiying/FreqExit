import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_wavelets import DWTForward, DWTInverse


_DWT = None
_IDWT = None
_DWT_DEVICE = None
_IDWT_DEVICE = None  

def get_dwt(device):
    global _DWT, _DWT_DEVICE
    if _DWT is None or _DWT_DEVICE != device:
        _DWT = DWTForward(J=1, mode='reflect', wave='haar').to(device)
        _DWT_DEVICE = device
    return _DWT

def get_idwt(device):
    global _IDWT, _IDWT_DEVICE  
    if _IDWT is None or _IDWT_DEVICE != device:
        _IDWT = DWTInverse(mode='reflect', wave='haar').to(device)
        _IDWT_DEVICE = device
    return _IDWT

class SubbandAlignmentModule(nn.Module):
   
    def __init__(self, channels=32):
        super(SubbandAlignmentModule, self).__init__()
        
        self.s_ll = nn.Parameter(torch.ones(channels))       
        self.s_lh = nn.Parameter(torch.full((channels,), 0.5))
        self.s_hl = nn.Parameter(torch.full((channels,), 0.5))
        self.s_hh = nn.Parameter(torch.zeros(channels))     
        
        self.conv1x1 = nn.Conv2d(4 * channels, channels, kernel_size=1, bias=True)
        
        self.inv_conv1x1 = nn.Conv2d(channels, 4 * channels, kernel_size=1, bias=True)
        
        with torch.no_grad():
           
            weight = torch.zeros(channels, 4 * channels, 1, 1)
            for i in range(channels):
                for j in range(4):
                    weight[i, i + j*channels, 0, 0] = 0.5  
            self.conv1x1.weight.copy_(weight)
            
          
            inv_weight = torch.zeros(4 * channels, channels, 1, 1)
            for i in range(channels):
                for j in range(4):
                    inv_weight[i + j*channels, i, 0, 0] = 0.5 
            self.inv_conv1x1.weight.copy_(inv_weight)
            
          
            if self.conv1x1.bias is not None:
                self.conv1x1.bias.zero_()
            if self.inv_conv1x1.bias is not None:
                self.inv_conv1x1.bias.zero_()
    
    def orthogonal_regularization(self):
      
      
        w = self.conv1x1.weight.view(self.conv1x1.weight.size(0), -1)
        w_t_w = torch.matmul(w, w.t())
        i_mat = torch.eye(w.size(0), device=w.device)
        orth_loss_olp = F.mse_loss(w_t_w, i_mat)
        
      
        w_inv = self.inv_conv1x1.weight.view(self.inv_conv1x1.weight.size(0), -1)
        w_inv_w_inv_t = torch.matmul(w_inv, w_inv.t())
        i_mat_inv = torch.eye(w_inv.size(0), device=w_inv.device)
        orth_loss_iolp = F.mse_loss(w_inv_w_inv_t, i_mat_inv)
       
        w_t = w.t()  
        conj_loss = F.mse_loss(w_inv, w_t)
        
        return orth_loss_olp + orth_loss_iolp + conj_loss
    
    def forward(self, x):
     
        B, C, H, W = x.shape
        device = x.device
        orig_dtype = x.dtype
        
     
        with torch.cuda.amp.autocast(enabled=False):
          
            dwt = get_dwt(device)
            idwt = get_idwt(device)
            
          
            pad_right, pad_bottom = W % 2, H % 2
            if pad_right or pad_bottom:
                mode = 'reflect' if min(H, W) > 1 else 'replicate'
                x_pad = F.pad(x, (0, pad_right, 0, pad_bottom), mode=mode)
            else:
                x_pad = x
            
          
            x_pad_fp32 = x_pad.float()
            
         
            coeffs = dwt(x_pad_fp32)
            
         
            LL = coeffs[0]  
            HF = coeffs[1][0]  
            LH = HF[:, :, 0]  
            HL = HF[:, :, 1]  
            HH = HF[:, :, 2]  
            
           
            LL_scaled = LL * torch.exp(self.s_ll)[None,:,None,None]
            LH_scaled = LH * torch.exp(self.s_lh)[None,:,None,None]
            HL_scaled = HL * torch.exp(self.s_hl)[None,:,None,None]
            HH_scaled = HH * torch.exp(self.s_hh)[None,:,None,None]
            
           
            subbands = torch.cat([LL_scaled, LH_scaled, HL_scaled, HH_scaled], dim=1)  
            
          
            low_proj = self.conv1x1(subbands)  
            
          
            restored = self.inv_conv1x1(low_proj)  
            
            
            chunks = restored.chunk(4, dim=1)
            ll_r, lh_r, hl_r, hh_r = chunks
            
            
            yl = ll_r  
            
            
            yh_tensor = torch.stack([lh_r, hl_r, hh_r], dim=2)  
            
        
            reconstructed = idwt((yl, [yh_tensor]))  
            
           
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
    orth_reg_weight=0.1
):
    

    with torch.cuda.amp.autocast(enabled=False):
        total_fgsr_loss = 0.0
        total_reg_loss = 0.0
        num_valid_layers = 0
        
      
        orig_dtype = None
        if student_token_maps and next(iter(student_token_maps.values())).numel() > 0:
            orig_dtype = next(iter(student_token_maps.values())).dtype
        
        
        step_indices = []
        cur_idx = 0
        for i, pn in enumerate(patch_nums):
            step_indices.append((cur_idx, cur_idx + pn*pn))
            cur_idx += pn*pn
        
       
        for layer_idx in supervise_layers:
            if layer_idx not in student_token_maps:
                continue
            
           
            mask = sample_masks[layer_idx]
            if mask is None or not mask.any():
                continue
            
            
            s_token_map = student_token_maps[layer_idx][mask]
            
           
            layer_fgsr_loss = 0.0
            step_weights_sum = 0.0
            
            for step_idx in all_steps:
                if step_idx >= len(patch_nums):
                    continue
                
                pn = patch_nums[step_idx]
                start_idx, end_idx = step_indices[step_idx]
                
                slice_feat = s_token_map[:, start_idx:end_idx]
                
                if slice_feat.shape[0] == 0:
                    continue
                
                vae_channels = slice_feat.shape[-1]
                r_t = slice_feat.transpose(1, 2).reshape(-1, vae_channels, pn, pn) 
                
                sam_module = sam_modules[f"step_{step_idx}"]
                r_aux = sam_module(r_t)
                
                mse_loss = F.mse_loss(r_aux, r_t)
                
                step_weight = 1.0  
                
                layer_fgsr_loss += step_weight * mse_loss
                step_weights_sum += step_weight
            
           
            layer_reg_loss = 0.0
            for step_idx in all_steps:
                if step_idx >= len(patch_nums):
                    continue
                module_key = f"step_{step_idx}"
                if module_key in sam_modules:
                    layer_reg_loss += sam_modules[module_key].orthogonal_regularization()
            
           
            if step_weights_sum > 0:
                layer_fgsr_loss = layer_fgsr_loss / step_weights_sum
                total_fgsr_loss += layer_fgsr_loss
                total_reg_loss += layer_reg_loss / len(all_steps)  
                num_valid_layers += 1
        
       
        if num_valid_layers > 0:
            avg_fgsr_loss = total_fgsr_loss / num_valid_layers
            avg_reg_loss = total_reg_loss / num_valid_layers
            final_fgsr_loss = avg_fgsr_loss + orth_reg_weight * avg_reg_loss
           
            if orig_dtype is not None:
                return final_fgsr_loss.to(orig_dtype)
            return final_fgsr_loss
        else:
            device = next(iter(sam_modules.values())).device
            if orig_dtype is not None:
                return torch.tensor(0.0, device=device, dtype=orig_dtype)
            return torch.tensor(0.0, device=device)


