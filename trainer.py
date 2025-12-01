import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.nn.functional as F
import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger
from utils.dloss import distillation_loss
from utils.hfloss import high_frequency_loss
from utils.fgsrloss import frequency_guided_self_reconstruction_loss, SubbandAlignmentModule
from utils.ema import EMA
from utils.loss_tracker import LossTracker
Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
import math

class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_student: VAR, var: DDP, var_teacher: VAR,
        var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(VARTrainer, self).__init__()
        
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp = var_student
        self.var_opt = var_opt

        self.var_teacher = var_teacher
        
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        
        self.vae_channels = vae_local.Cvae
        self.sam_modules = nn.ModuleDict()
        
        for step_idx, pn in enumerate(self.patch_nums):
            module = SubbandAlignmentModule(channels=self.vae_channels).to(device)
            self.sam_modules[f"step_{step_idx}"] = module
            

        print(f"Initialized {len(self.patch_nums)} step-specific subband alignment modules")
        
        self.ema = EMA(self.var_wo_ddp, decay=0.9999)
        print("Model EMA initialized with decay rate=0.9999")
        
        self.loss_tracker = LossTracker(momentum=0.9)
        
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        self.ema.store()
        self.ema.copy_to_model()
        
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            self.var_wo_ddp.forward
            outputs = self.var(label_B, x_BLCv_wo_first_l, 0)
            logits_BLV = outputs["logits"]

            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        
        self.ema.restore()
        self.var_wo_ddp.train(training)
        
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    

    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float, max_it: int = None, 
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1
        if prog_si == len(self.patch_nums) - 1: prog_si = -1
        
        epoch_ratio = g_it / max_it if max_it else 0.0
        
        apply_hf_loss = False

        if epoch_ratio < 0.25:
            hf_loss_weight = 0.0
            apply_hf_loss = False
        elif epoch_ratio < 0.5:
            hf_loss_weight = 0.3 * ((epoch_ratio - 0.25) / 0.25)
            apply_hf_loss = True
        else:
            hf_loss_weight = 0.3 + 0.1 * ((epoch_ratio - 0.5) / 0.5)
        
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        
        with self.var_opt.amp_ctx:
            teacher_logits_BLV = self.var_teacher.forward_orig(label_B, x_BLCv_wo_first_l)
            outputs = self.var(label_B, x_BLCv_wo_first_l, g_it)
            
            if not isinstance(outputs, dict) or "all_logits" not in outputs:
                raise ValueError("Model output format error, unable to get logits")
            
            logits_BLV = outputs['logits']
            all_logits = outputs['all_logits']
            supervise_mask = outputs.get('supervise_mask', [False] * self.var_wo_ddp.depth)
            sample_masks = outputs.get('sample_masks', [])
            features = outputs.get('features', {})
            
            supervise_layers_with_samples = []
            for layer_idx in all_logits:
                if layer_idx < len(sample_masks) and sample_masks[layer_idx] is not None and sample_masks[layer_idx].any():
                    supervise_layers_with_samples.append(layer_idx)
            
            if prog_si >= 0:
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:
                lw = self.loss_weight
            
            main_ce_loss_raw = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            main_ce_loss = main_ce_loss_raw.mul(lw).sum(dim=-1).mean()
            
            early_exit_loss = 0.0
            layer_losses = {}
            
            if supervise_layers_with_samples:
                normalization_factor = sum(
                    self.var_wo_ddp.layer_weights[layer_idx] 
                    for layer_idx in supervise_layers_with_samples
                )
                
                for layer_idx in supervise_layers_with_samples:
                    layer_logits = all_logits[layer_idx]
                    
                    layer_loss_raw = self.train_loss(layer_logits.view(-1, V), gt_BL.view(-1)).view(B, -1)
                    
                    if layer_idx >= len(sample_masks) or sample_masks[layer_idx] is None:
                        raise ValueError(f"Layer {layer_idx} sample mask does not exist or is None, dropout application error")
                    
                    sample_mask = sample_masks[layer_idx]
                    if not sample_mask.any():
                        raise ValueError(f"Layer {layer_idx} has no samples processed (all samples masked as False), dropout application error")
                    
                    per_sample_loss = layer_loss_raw.mul(lw).sum(dim=-1)
                    
                    processed_loss = per_sample_loss[sample_mask].mean()
                    
                    normalized_weight = self.var_wo_ddp.layer_weights[layer_idx] / normalization_factor
                    early_exit_loss += processed_loss * normalized_weight
                    
                    layer_losses[f"layer_{layer_idx}_loss"] = processed_loss.item()
            
            distill_loss = distillation_loss(
                teacher_logits=teacher_logits_BLV,
                student_logits_dict=all_logits,
                supervise_layers=supervise_layers_with_samples,
                student_layer_weights=self.var_wo_ddp.layer_weights,
                sample_masks=sample_masks
            )

            vae_embedding = self.vae_local.quantize.embedding.weight
            teacher_probs_all = F.softmax(teacher_logits_BLV.detach(), dim=-1)
            teacher_token_map_all = torch.matmul(teacher_probs_all, vae_embedding)
            
            student_token_maps = {}
            
            for layer_idx in supervise_layers_with_samples:
                if layer_idx >= len(sample_masks) or sample_masks[layer_idx] is None:
                    continue
                    
                mask = sample_masks[layer_idx]
                if not mask.any():
                    continue
                    
                if layer_idx not in all_logits:
                    continue
                
                student_logits = all_logits[layer_idx]
                student_probs = F.softmax(student_logits, dim=-1)
                student_token_maps[layer_idx] = torch.matmul(student_probs, vae_embedding)
            
            if apply_hf_loss:
                hf_loss = high_frequency_loss(
                    teacher_token_map=teacher_token_map_all,
                    student_token_maps=student_token_maps,
                    supervise_layers=supervise_layers_with_samples,
                    sample_masks=sample_masks,
                    patch_nums=self.patch_nums,
                    hf_steps=[5, 6, 7, 8, 9],
                    student_layer_weights=self.var_wo_ddp.layer_weights,
                )
                
                self.loss_tracker.update(main_ce_loss, early_exit_loss, distill_loss, hf_loss)
                hf_scale = self.loss_tracker.get_hf_scale(alpha=0.3)
                hf_loss = hf_loss * hf_scale
            else:
                hf_loss = torch.tensor(0.0, device=main_ce_loss.device)
            
            fgsr_loss = frequency_guided_self_reconstruction_loss(
                student_token_maps=student_token_maps,
                sample_masks=sample_masks,
                supervise_layers=supervise_layers_with_samples,
                sam_modules=self.sam_modules,
                patch_nums=self.patch_nums,
                all_steps=list(range(10)),
                orth_reg_weight=0.1
            )
            
            main_loss_weight = 0.1
            early_exit_weight = 0.5
            distill_weight = 2.0
            fgsr_loss_weight = 5.0
            
            if not apply_hf_loss:
                hf_loss_weight = 0.0
            else:
                hf_loss_weight = 0.3

            loss = (
                main_loss_weight * main_ce_loss + 
                early_exit_weight * early_exit_loss + 
                distill_weight * distill_loss +
                hf_loss_weight * hf_loss +
                fgsr_loss_weight * fgsr_loss
            )

            weighted_loss_dict = {
                'main_ce_loss': main_loss_weight * main_ce_loss,
                'early_exit_loss': early_exit_weight * early_exit_loss,
                'distill_loss': distill_weight * distill_loss,
                'hf_loss': hf_loss_weight * hf_loss,
                'fgsr_loss': fgsr_loss_weight * fgsr_loss
            }

        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        if stepping:
            self.ema.update()
        
        pred_BL = logits_BLV.data.argmax(dim=-1)

        if (it == 0 or it in metric_lg.log_iters):
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:
                Ltail = acc_tail = -1
            else:
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100

            early_exit_loss_val = early_exit_loss.item() if supervise_layers_with_samples else 0.0
            main_loss_val = main_ce_loss.item()
            combined_loss_val = loss.item()
            
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, 
                            EE=early_exit_loss_val, ML=main_loss_val, CL=combined_loss_val)
            
            if stepping and grad_norm is not None:
                metric_lg.update(tnm=grad_norm.item())

        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
                
                tb_update_dict = {
                    'early_exit_loss': early_exit_loss.item() if supervise_layers_with_samples else 0.0,
                    'main_loss': main_ce_loss.item(),
                    'combined_loss': loss.item(),
                    'distill_loss': distill_loss.item(),
                    'hf_loss': hf_loss.item(),
                    'fgsr_loss': fgsr_loss.item(),
                }
                
                tb_lg.update(head='AR_loss_components', **tb_update_dict, step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        
        state['ema'] = self.ema.state_dict()
        
        return state

    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        if 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            print(f'[VARTrainer.load_state_dict] EMA state loaded')
        else:
            print(f'[VARTrainer.load_state_dict] EMA state not found, will initialize EMA with current model parameters')
            self.ema.register()
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
