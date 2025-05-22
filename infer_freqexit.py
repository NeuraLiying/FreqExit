
import os
import os.path as osp
import torch, torchvision
import torch.nn.functional as F
import random
import numpy as np
import PIL.Image as PImage
import argparse
import sys
import time
import statistics
from torch.profiler import profile, record_function, ProfilerActivity
from typing import List, Dict
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description='VAR image generation and layer skipping performance test')
parser.add_argument('--model_depth', type=int, default=20, choices=[16, 20, 24, 30, 36], help='model depth')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--cfg', type=float, default=1.5, help='classifier guidance strength')
parser.add_argument('--pn', type=str, default='256', choices=['256', '512', '1024'], help='patch number')
parser.add_argument('--more_smooth', action='store_true', help='more smooth output')
parser.add_argument('--output_dir', type=str, default='outputs_layerskip_8', help='output directory')
parser.add_argument('--profile_flops', type=bool, default=True, help='whether to analyze FLOPS')

parser.add_argument('--exit_config', type=int, default=None, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8], 
                    help='select layer skip config')

parser.add_argument('--samples_per_class', type=int, default=50, help='number of samples per class')
parser.add_argument('--total_classes', type=int, default=1000, help='total number of classes to generate')
parser.add_argument('--fid_output_dir', type=str, default=None, required=True, help='FID sample output directory')
args = parser.parse_args()


setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from models import build_vae_var

MODEL_DEPTH = args.model_depth
assert MODEL_DEPTH in {16, 20, 24, 30, 36}


vae_ckpt = 'path_to_vae_checkpoint'
var_ckpt = 'path_to_var_checkpoint'


if args.pn == '256':
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,  
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
)


print("Loading model weights...")
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

checkpoint = torch.load(var_ckpt, map_location='cpu')
if 'trainer' in checkpoint:
    print("Detected training checkpoint file, extracting model weights...")
    if 'var_wo_ddp' in checkpoint['trainer']:
        model_weights = checkpoint['trainer']['var_wo_ddp']
        var.load_state_dict(model_weights, strict=True)
        print("Successfully extracted model weights from training checkpoint")
    else:
        print("Warning: var_wo_ddp not found in checkpoint, trying direct loading...")
else:
    print("Warning: Not using training checkpoint file")
    var.load_state_dict(checkpoint, strict=True)

vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'Model preparation completed')


seed = args.seed
cfg = args.cfg
more_smooth = args.more_smooth

print(f"Using parameters: model_depth={MODEL_DEPTH}, seed={seed}, CFG strength={cfg}")
print(f"Smooth mode: {'enabled' if more_smooth else 'disabled'}")


torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_


os.makedirs(args.output_dir, exist_ok=True)

def parse_exit_layers(args):
   
    exit_configs = {
        

        1: [20,20,20,20,20,20,20,20,20,20],
        
        2: [16,16,16,16,16,16,16,16,16,16],

        3: [14,14,14,14,14,14,14,14,14,14],
        
        4: [12,12,12,12,12,12,12,12,12,12],
        
        
        5:  [10,10,10,10,10,10,10,10,10,10],
        

    }
    
   
    if args.exit_config in exit_configs:
        return exit_configs[args.exit_config]
    else:
       
        return exit_configs[1]

def generate_image_with_layerskip(class_id, image_idx, base_seed, exit_layers_per_step):
   
    
  
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    

    cur_seed = base_seed + image_idx
    torch.manual_seed(cur_seed)
    random.seed(cur_seed)
    np.random.seed(cur_seed)
    
    exit_layers = []
    
    label_B = torch.tensor([class_id], device=device)
    
    start_time = time.time()
    
    memory_samples = []
    
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            
            var.rng.manual_seed(cur_seed)
            rng = var.rng
            
            sos = cond_BD = var.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=var.num_classes)), dim=0))
            
            lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
            
            next_token_map = sos.unsqueeze(1).expand(2, var.first_l, -1) + var.pos_start.expand(2, var.first_l, -1) + lvl_pos[:, :var.first_l]
            
            cur_L = 0
            f_hat = sos.new_zeros(1, var.Cvae, var.patch_nums[-1], var.patch_nums[-1])
            
            for b in var.blocks: 
                b.attn.kv_caching(True)
            
            quant_resi = var.vae_quant_proxy[0].quant_resi
            
            for si, pn in enumerate(var.patch_nums):

                ratio = si / var.num_stages_minus_1

                cur_L += pn*pn
                

                if si < len(exit_layers_per_step):
                    exit_layer = exit_layers_per_step[si]
                else:
                    exit_layer = exit_layers_per_step[-1]
                
              
                cond_BD_or_gss = var.shared_ada_lin(cond_BD)
                
            
                x = next_token_map
                
             
                for i in range(min(exit_layer, var.depth)):
                    x = var.blocks[i](x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
             
                exit_layers.append(exit_layer)
                
              
                logits_BlV = var.get_logits(x, cond_BD)
                
             
                t = cfg * ratio
             
                logits_BlV = (1+t) * logits_BlV[:1] - t * logits_BlV[1:]
                
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=900, top_p=0.95, num_samples=1)[:, :, 0]
                
                if not more_smooth:
            
                    h_BChw = var.vae_quant_proxy[0].embedding(idx_Bl)
                else:
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                    h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ var.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                
                h_BChw = h_BChw.transpose_(1, 2).reshape(1, var.Cvae, pn, pn)
                
                HW = var.patch_nums[-1]  
                if si != len(var.patch_nums)-1:
                    h = quant_resi[si/(len(var.patch_nums)-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))
                    f_hat.add_(h)  
                    next_token_map = F.interpolate(f_hat, size=(var.patch_nums[si+1], var.patch_nums[si+1]), mode='area')
                else:
                   
                    h = quant_resi[si/(len(var.patch_nums)-1)](h_BChw)
                    f_hat.add_(h)
                    next_token_map = f_hat
                
               
                if si != var.num_stages_minus_1:
                    next_token_map = next_token_map.view(1, var.Cvae, -1).transpose(1, 2)
                    next_token_map = var.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var.patch_nums[si+1] ** 2]
                    next_token_map = next_token_map.repeat(2, 1, 1)
                
             
                if torch.cuda.is_available():
                    mem_sample = torch.cuda.memory_allocated() / (1024 * 1024)  
            
        
            for b in var.blocks: 
                b.attn.kv_caching(False)
            
           
            recon_B3HW = var.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
    
 
    end_time = time.time()
    latency = end_time - start_time  
    throughput = 1 / latency  
    
   
    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  
    else:
        memory = 0
    
   
    output_dir = os.path.join(args.output_dir, args.fid_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    img = recon_B3HW[0].permute(1, 2, 0).mul(255).cpu().numpy()
    img = PImage.fromarray(img.astype(np.uint8))
    img.save(f'{output_dir}/var_d{MODEL_DEPTH}_class{class_id}_idx{image_idx}.png')
    
    return {
        'latency': latency,
        'throughput': throughput,
        'memory': memory,
        'exit_layers': exit_layers,
        'avg_exit_layer': statistics.mean(exit_layers)
    }

def measure_layerskip_flops(exit_layers_per_step):
    print("\nMeasuring FLOPS...")
    
  
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("var_model_inference"):
            generate_image_with_layerskip(0, 0, args.seed, exit_layers_per_step)
    
   
    total_flops = 0
    for event in prof.key_averages():
        if event.flops > 0:
            total_flops += event.flops
    
    total_params = 0
    for name, param in vae.named_parameters():
        total_params += param.numel()
    for name, param in var.named_parameters():
        total_params += param.numel()
    
    total_gflops = total_flops / 1e9
    print(f"\n===== FLOPS Statistics (Terminal Display) =====")
    print(f"Total FLOPS: {total_gflops:.2f} GFLOPs")
    print(f"Model parameters: {total_params/1e6:.2f} M params")
    print(f"Average FLOPS per layer: {total_gflops/MODEL_DEPTH:.2f} GFLOPs")
    
    avg_exit_layer = sum(exit_layers_per_step) / len(exit_layers_per_step)
    saved_ratio = (MODEL_DEPTH - avg_exit_layer) / MODEL_DEPTH
    print(f"Average exit layer: {avg_exit_layer:.2f}/{MODEL_DEPTH}")
    print(f"Layer skip savings: {saved_ratio*100:.2f}% FLOPS")
    
    return {
        'total_flops': total_gflops, 
        'total_params': total_params/1e6,
        'avg_exit_layer': avg_exit_layer,
        'saved_ratio': saved_ratio
    }

if __name__ == "__main__":
    exit_layers_per_step = parse_exit_layers(args)
    config_names = {
        1: "All 20 layers (avg 20 layers)",
        2: "First half 20 layers, second half 16 layers (avg 18 layers)",
        3: "All 16 layers (avg 16 layers)", 
        4: "First half 16 layers, second half 12 layers (avg 14 layers)",
        5: "First half 16 layers, second half 8 layers (avg 12 layers)",
        6: "All 12 layers (avg 12 layers)",
        7: "First half 12 layers, second half 8 layers (avg 10 layers)",
        8: "All 10 layers (avg 10 layers)"
    }
    print(f"Using layer skip config {args.exit_config}: {config_names[args.exit_config]}")
    print(f"Specific config: {exit_layers_per_step}")
    
    output_dir = os.path.join(args.output_dir, args.fid_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"FID evaluation mode: Generating {args.total_classes} classes, {args.samples_per_class} images per class")
    total_images = args.total_classes * args.samples_per_class
    
    flops_data = None
    if args.profile_flops:
        flops_data = measure_layerskip_flops(exit_layers_per_step)
    
    latency_records = []  
    throughput_records = []  
    memory_records = []  
    exit_layer_records = []  
    
    progress_bar = tqdm(total=total_images, desc="Generating images")
    
    running_latency = 0
    running_throughput = 0
    running_memory = 0
    running_exit_layer = 0
    image_count = 0
    
    for class_id in range(args.total_classes):
        for img_idx in range(args.samples_per_class):
            stats = generate_image_with_layerskip(class_id, img_idx, args.seed, exit_layers_per_step)
            
            latency_records.append(stats['latency'])
            throughput_records.append(stats['throughput'])
            memory_records.append(stats['memory'])
            exit_layer_records.append(stats['avg_exit_layer'])
            
            image_count += 1
            running_latency = statistics.mean(latency_records)
            running_throughput = statistics.mean(throughput_records)
            running_memory = statistics.mean(memory_records)
            running_exit_layer = statistics.mean(exit_layer_records)
            
            desc = f"Latency: {running_latency:.4f}s/img | Throughput: {running_throughput:.2f}img/s | Memory: {running_memory:.2f}MB | Avg exit layer: {running_exit_layer:.2f}/{MODEL_DEPTH}"
            progress_bar.set_description(desc)
            progress_bar.update(1)
    
    progress_bar.close()
    
    avg_latency = statistics.mean(latency_records)
    std_latency = statistics.stdev(latency_records) if len(latency_records) > 1 else 0
    avg_throughput = statistics.mean(throughput_records)
    std_throughput = statistics.stdev(throughput_records) if len(throughput_records) > 1 else 0
    avg_memory = statistics.mean(memory_records)
    avg_exit_layer = statistics.mean(exit_layer_records)
    std_exit_layer = statistics.stdev(exit_layer_records) if len(exit_layer_records) > 1 else 0
    
    computation_saved = (MODEL_DEPTH - avg_exit_layer) / MODEL_DEPTH * 100
    
    print("\n========== VAR-LayerSkip Performance Test Results ==========")
    print(f"Model depth: {MODEL_DEPTH}")
    print(f"Generated images: {image_count}")
    print(f"Layer skip config {args.exit_config}: {config_names[args.exit_config]}")
    print(f"Specific config: {exit_layers_per_step}")
    
    print("\n1. Latency:")
    print(f"   Average: {avg_latency:.6f} seconds/image")
    print(f"   Std dev: {std_latency:.6f} seconds")
    
    print("\n2. Throughput:")
    print(f"   Average: {avg_throughput:.4f} images/second")
    print(f"   Std dev: {std_throughput:.4f}")
    
    print("\n3. Memory usage:")
    print(f"   Average memory: {avg_memory:.2f} MB")
    
    print("\n4. LayerSkip statistics:")
    print(f"   Average exit layer: {avg_exit_layer:.2f} / {MODEL_DEPTH}")
    print(f"   Std dev: {std_exit_layer:.2f}")
    print(f"   Computation saved: {computation_saved:.2f}%")
    
    if args.profile_flops and flops_data:
        print("\n5. Computation efficiency:")
        print(f"   Total FLOPS: {flops_data['total_flops']:.2f} GFLOPs")
        print(f"   Average FLOPS per layer: {flops_data['total_flops']/MODEL_DEPTH:.2f} GFLOPs")
        print(f"   Early exit average saved layers: {MODEL_DEPTH - avg_exit_layer:.2f}/{MODEL_DEPTH}")
        saved_flops = (MODEL_DEPTH - avg_exit_layer)/MODEL_DEPTH * flops_data['total_flops'] 
        print(f"   Early exit saved FLOPS: {saved_flops:.2f} GFLOPs ({computation_saved:.2f}%)")
    
    results_file = os.path.join(args.output_dir, f"layerskip_results_d{MODEL_DEPTH}.txt")
    with open(results_file, 'w') as f:
        f.write("========== VAR-LayerSkip Performance Test Results ==========\n")
        f.write(f"Model depth: {MODEL_DEPTH}\n")
        f.write(f"Generated images: {image_count}\n")
        f.write(f"Layer skip config {args.exit_config}: {config_names[args.exit_config]}\n")  
        f.write(f"Specific config: {exit_layers_per_step}\n\n")
        
        f.write("1. Latency:\n")
        f.write(f"   Average: {avg_latency:.6f} seconds/image\n")
        f.write(f"   Std dev: {std_latency:.6f} seconds\n\n")
        
        f.write("2. Throughput:\n")
        f.write(f"   Average: {avg_throughput:.4f} images/second\n")
        f.write(f"   Std dev: {std_throughput:.4f}\n\n")
        
        f.write("3. Memory usage:\n")
        f.write(f"   Average memory: {avg_memory:.2f} MB\n\n")
        
        f.write("4. LayerSkip statistics:\n")
        f.write(f"   Average exit layer: {avg_exit_layer:.2f} / {MODEL_DEPTH}\n")
        f.write(f"   Std dev: {std_exit_layer:.2f}\n")
        f.write(f"   Computation saved: {computation_saved:.2f}%\n")
        
        if args.profile_flops and flops_data:
            f.write("\n5. Computation efficiency:\n")
            f.write(f"   Total FLOPS: {flops_data['total_flops']:.2f} GFLOPs\n")
            f.write(f"   Average FLOPS per layer: {flops_data['total_flops']/MODEL_DEPTH:.2f} GFLOPs\n")
            f.write(f"   Early exit average saved layers: {MODEL_DEPTH - avg_exit_layer:.2f}/{MODEL_DEPTH}\n")
            saved_flops = (MODEL_DEPTH - avg_exit_layer)/MODEL_DEPTH * flops_data['total_flops']
            f.write(f"   Early exit saved FLOPS: {saved_flops:.2f} GFLOPs ({computation_saved:.2f}%)\n")
            
    print(f"Performance test results saved to: {results_file}")
    print("Done!")