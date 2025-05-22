import torch
import torch.nn as nn
import torch.nn.functional as F



def distillation_loss(
    teacher_logits, 
    student_logits_dict,
    sample_masks, 
    supervise_layers, 
    student_layer_weights,
    T_max=4.0,  
    T_min=1.0,   
    student_final_layer_idx=19  
):
    
    if len(supervise_layers) == 0:
        return torch.tensor(0.0, device=teacher_logits.device)
    
    total_loss = 0.0
    total_weight = 0.0
    teacher = teacher_logits.detach()  
    
    for layer_idx in supervise_layers:
        if layer_idx >= len(sample_masks) or sample_masks[layer_idx] is None:
            continue
            
        mask = sample_masks[layer_idx]
        if not mask.any():
            continue
            
        if layer_idx not in student_logits_dict:
            continue

       
        ratio = layer_idx / student_final_layer_idx
        T = T_max - (T_max - T_min) * ratio

       
        stu = student_logits_dict[layer_idx][mask]
        tea = teacher[mask]
        stu = stu.view(-1, stu.size(-1))
        tea = tea.view(-1, tea.size(-1))

        
        soft_t = F.softmax(tea / T, dim=-1)
        log_s = F.log_softmax(stu / T, dim=-1)
        kl = F.kl_div(log_s, soft_t, reduction='none') * (T*T)
        
       
        kl = kl.sum(dim=-1).mean()  

       
        w = student_layer_weights[layer_idx]
        total_loss += w * kl
        total_weight += w
    
    if total_weight > 0:
        return total_loss / total_weight
    else:
        return torch.tensor(0.0, device=teacher.device)