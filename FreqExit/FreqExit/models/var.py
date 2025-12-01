import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()

    
        
       
        assert embed_dim % num_heads == 0
       
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size  
       
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads  
        
      
        self.cond_drop_rate = cond_drop_rate
   
        self.prog_si = -1   
        
       
        self.patch_nums: Tuple[int] = patch_nums  
     
        self.L = sum(pn ** 2 for pn in self.patch_nums)  
        self.first_l = self.patch_nums[0] ** 2 
        self.begin_ends = []
        
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1  
        self.rng = torch.Generator(device=dist.get_device())
        
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)  
        
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes  
        
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())  
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)  
        
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))  
        
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
        
            pe = torch.empty(1, pn*pn, self.C)  
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        
        pos_1LC = torch.cat(pos_1LC, dim=1)     
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)  
        
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)  
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)  
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)  
        dT = d.transpose(1, 2)    
        lvl_1L = dT[:, 0].contiguous()  
        self.register_buffer('lvl_1L', lvl_1L)
        
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)  
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)  

 
        self.num_layers = depth
        self.p_max = 0.1
        self.layer_dropout = True
        self.d_layer = []

        if self.num_layers == 1:
            self.d_layer = [0.0]
        else:
            for l in range(self.num_layers):
                if l == 0:
                 
                    self.d_layer.append(0.0)
                else:
                  
                    ratio = (l * math.log(2)) / (self.num_layers - 1)
                    val = math.exp(ratio) - 1.0
                    val = max(0.0, min(val, 1.0))  
                    self.d_layer.append(val)

        self.R = 4
        self.escale = 1
        self.apply_early_exit_loss = True
        self.layer_weights = []

        for l in range(self.depth):
            if l <self.depth - 1:
                weight = self.escale * sum(i for i in range(l + 1))
            else:
                weight = (self.depth - 1) + self.escale * sum(i for i in range(self.depth - 1))
            
            self.layer_weights.append(weight)

     
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
   
        if not isinstance(h_or_h_and_residual, torch.Tensor):
     
            h, resi = h_or_h_and_residual   
 
            h = resi + self.blocks[-1].drop_path(h)  
        else:
            h = h_or_h_and_residual  
        
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   
           
    
        if g_seed is None: 
            rng = None
        else: 
       
            self.rng.manual_seed(g_seed)
            rng = self.rng
        
        if label_B is None:
           
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
         
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
   
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        
       
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        

        for b in self.blocks: 
            b.attn.kv_caching(True)
        
        
        for si, pn in enumerate(self.patch_nums):   
           
            ratio = si / self.num_stages_minus_1  
           
            cur_L += pn*pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            
            x = next_token_map  
            
            for b in self.blocks:
                
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            
            
            logits_BlV = self.get_logits(x, cond_BD)
            
            
            t = cfg * ratio
            
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            
            if not more_smooth:  
            
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            else:   
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   

                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            
            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)
        
        for b in self.blocks: 
            b.attn.kv_caching(False)
        
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, step: int = None) -> dict:
       
        MASK_SEED_BASE = 42
        
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        
      
        with torch.cuda.amp.autocast(enabled=False):
           
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
           
            sos = cond_BD = self.class_emb(label_B)
            
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: 
                x_BLC = sos  
            else: 
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        batch_size = x_BLC.shape[0]
        device = x_BLC.device  
 
        AdaLNSelfAttn.forward
        all_hidden = {}

        supervised_layers = []
        if self.training and self.apply_early_exit_loss:
            min_layer = 0  
            max_layer = self.depth - 1

            min_layer = max(0, min(min_layer, self.depth - 1))
            max_layer = max(min_layer, min(max_layer, self.depth - 1))

            valid_layers = max_layer - min_layer + 1

            block_size = max(1, (valid_layers + self.R - 1) // self.R)
            relative_offset = (step % self.R) * block_size
            offset = min_layer + relative_offset

            if offset >= self.depth:
                offset = min_layer

            end_layer_idx = min(offset + block_size, self.depth)
            supervised_layers = list(range(offset, end_layer_idx))

            supervise_mask = [False] * self.depth
            for l in supervised_layers:
                supervise_mask[l] = True

        else:

            supervised_layers = [self.depth - 1]
            supervise_mask = [False] * self.depth
            supervise_mask[self.depth - 1] = True

        self.sample_masks = []

        for i, b in enumerate(self.blocks):
            original_input = x_BLC
            
            if self.training and self.layer_dropout and i > 0:  
                p_l = self.d_layer[i] * self.p_max
                
                drop_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                
                is_distributed = dist.initialized()
                rank = 0
                if is_distributed:
                    rank = dist.get_rank()
                
                drop_count = 0
                if rank == 0:
               
                    generator = torch.Generator(device=device)
                    generator.manual_seed(int(MASK_SEED_BASE + step + i * 1000))
                    temp_mask = torch.rand(batch_size, device=device, generator=generator) < p_l
                    drop_count = temp_mask.sum().item()
                
                if is_distributed:
                    drop_count_tensor = torch.tensor([drop_count], device=device)
                    dist.broadcast(drop_count_tensor, src_rank=0)
                    drop_count = int(drop_count_tensor.item())
                
                drop_count = min(drop_count, batch_size - 1)
                
                if drop_count > 0:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(int(MASK_SEED_BASE + step + i * 1000))
                    
                    perm = torch.randperm(batch_size, device=device, generator=generator)
                    indices_to_drop = perm[:drop_count]
                    drop_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    drop_mask[indices_to_drop] = True
                
                keep_mask = ~drop_mask

                self.sample_masks.append(keep_mask)
                
                if keep_mask.any():  
                    keep_input = original_input[keep_mask]
                    keep_cond = cond_BD_or_gss[keep_mask] if cond_BD_or_gss is not None else None
                    
                    processed = b(x=keep_input, cond_BD=keep_cond, attn_bias=attn_bias)
                    
                    result = original_input.clone()
                    result[keep_mask] = processed
                    x_BLC = result
                else:
                    x_BLC = original_input
            else:

                x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)

                self.sample_masks.append(torch.ones(batch_size, dtype=torch.bool, device=device))

            if supervise_mask[i] or i == self.depth - 1:
                all_hidden[i] = x_BLC.clone()

            
        all_logits = {}
        for layer_idx in all_hidden:
            all_logits[layer_idx] = self.get_logits(all_hidden[layer_idx].float(), cond_BD)


        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                for layer_idx in all_logits:
                    all_logits[layer_idx][0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                for layer_idx in all_logits:
                    all_logits[layer_idx][0, 0, 0] += s

        return {
            "logits": all_logits[self.depth - 1],
            "all_logits": all_logits,
            "features": all_hidden,
            "supervise_mask": supervise_mask,
            "sample_masks": self.sample_masks
        }
    

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'
    

    def forward_orig(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
          
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        
        with torch.cuda.amp.autocast(enabled=False):
           
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
           
            sos = cond_BD = self.class_emb(label_B)
                
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
      
            if self.prog_si == 0: 
            
                x_BLC = sos  
            else: 
             
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
        
      
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
   
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        
        return x_BLC    
    
    @torch.no_grad()
    def autoregressive_infer_inpaint(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, f_hats=None, mask_in=None, mask_out=None
    ) -> torch.Tensor:  

        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        mask_in = F.interpolate(mask_in, size=(self.patch_nums[-1],self.patch_nums[-1]), mode='nearest')
        mask_out = F.interpolate(mask_out, size=(self.patch_nums[-1],self.patch_nums[-1]), mode='nearest')

        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):  


            ratio = si / self.num_stages_minus_1
            cur_L += pn*pn
            x = next_token_map

            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            if not more_smooth:  
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   
            else:   
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input_inpaint(si, len(self.patch_nums), f_hat, h_BChw, f_hats, mask_in, mask_out)

            if si != self.num_stages_minus_1:   
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   
 
        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   


class VARHF(VAR, PyTorchModelHubMixin):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
