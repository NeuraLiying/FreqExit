import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage
from PIL import Image, ImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from models import VQVAE, build_vae_var
import gc
from contextlib import contextmanager
import argparse
import torch.nn.functional as F

@contextmanager
def measure_peak_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    yield
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f'memory consumption: {peak_memory:.2f} MB')

def generate_random_mask(width=256, height=256, mask_type="rectangle", num_shapes=1):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    if mask_type == "rectangle":
        for _ in range(num_shapes):
            rect_w = int(width * random.uniform(0.2, 0.5))
            rect_h = int(height * random.uniform(0.2, 0.5))
            
            x1 = random.randint(0, width - rect_w)
            y1 = random.randint(0, height - rect_h)
            x2 = x1 + rect_w
            y2 = y1 + rect_h
            
            draw.rectangle([x1, y1, x2, y2], fill=255)
            
    elif mask_type == "circle":
        for _ in range(num_shapes):
            radius = int(width * random.uniform(0.15, 0.35))
            
            cx = random.randint(radius, width - radius)
            cy = random.randint(radius, height - radius)
            
            draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=255)
            
    elif mask_type == "random":
        points = []
        num_points = random.randint(6, 12)
        
        for _ in range(num_points):
            x = random.randint(width//4, width*3//4)
            y = random.randint(height//4, height*3//4)
            points.append((x, y))
        
        draw.polygon(points, fill=255)
    
    return mask

parser = argparse.ArgumentParser()
parser.add_argument("--model_depth", type=int, default=20)
parser.add_argument("--cfg", type=float, default=1.5)
parser.add_argument("--input_image", type=str, default='path_to_original_image', help="input image path to repair")
parser.add_argument("--output_image", type=str, default="output_inpaint.png", help="output image path")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--mask_type", type=str, default="rectangle", choices=["rectangle", "circle", "random"], help="mask type")
parser.add_argument("--num_shapes", type=int, default=1, help="number of shapes in mask")
parser.add_argument("--class_label", type=int, default=None, help="class label for generation")
parser.add_argument("--output_dir", type=str, default="outputs", help="root path for output directory")
args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

input_basename = os.path.basename(args.input_image)
input_name_without_ext = os.path.splitext(input_basename)[0]

output_dir = os.path.join(args.output_dir, f"class_{args.class_label}_seed_{args.seed}")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

MODEL_DEPTH = 20

vae_ckpt = 'path_to_vae_checkpoint'
var_ckpt = 'path_to_var_checkpoint'

patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

vae.load_state_dict(torch.load(vae_ckpt, map_location=device), strict=True)

checkpoint = torch.load(var_ckpt, map_location='cpu')
if 'trainer' in checkpoint:
    print("Detected training checkpoint file, extracting model weights...")
    if 'var_wo_ddp' in checkpoint['trainer']:
        model_weights = checkpoint['trainer']['var_wo_ddp']
        var.load_state_dict(model_weights, strict=True)
        print("Successfully extracted model weights from training checkpoint")
    else:
        print("Warning: var_wo_ddp not found in checkpoint, trying direct loading...")
        var.load_state_dict(checkpoint, strict=True)
else:
    print("Loading original model weights...")
    var.load_state_dict(checkpoint, strict=True)

vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'Preparation finished.')

input_image = Image.open(args.input_image).convert('RGB')
input_image = input_image.resize((256, 256))
input_tensor = torchvision.transforms.ToTensor()(input_image).unsqueeze(0).to(device)
input_tensor = input_tensor * 2.0 - 1.0

mask_image = generate_random_mask(
    width=256, 
    height=256, 
    mask_type=args.mask_type,
    num_shapes=args.num_shapes
)
mask_tensor = torchvision.transforms.ToTensor()(mask_image).unsqueeze(0).to(device)

mask_tensor = (mask_tensor > 0.5).float()

mask_in = 1.0 - mask_tensor
mask_out = mask_tensor

with torch.no_grad():
    f = vae.get_f(input_tensor)
    f_max_res = F.interpolate(f, size=(var.patch_nums[-1], var.patch_nums[-1]), mode='bicubic')
    f_hats = [f_max_res] * len(var.patch_nums)

masked_input = input_tensor.clone()
masked_input = masked_input * mask_in
masked_input = masked_input.add(1).mul(0.5)
masked_img = masked_input[0].permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
masked_img = PImage.fromarray(masked_img)

cfg = args.cfg
class_labels = [args.class_label]
B = len(class_labels)
more_smooth = False

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.inference_mode():
    label_B = torch.tensor(class_labels, device=device)
    with measure_peak_memory():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            start_event.record()
            recon_B3HW = var.autoregressive_infer_inpaint(
                B=B, 
                label_B=label_B, 
                cfg=cfg, 
                top_k=900, 
                top_p=0.95, 
                g_seed=seed, 
                more_smooth=more_smooth,
                f_hats=f_hats,
                mask_in=mask_in,
                mask_out=mask_out
            )
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            print("running time:",int(elapsed_time),"ms", "batch size:",str(len(class_labels)))

    inpaint_filename = f"{input_name_without_ext}_inpaint.png"
    comparison_filename = f"{input_name_without_ext}_comparison.png"
    
    inpaint_path = os.path.join(output_dir, inpaint_filename)
    output_img = recon_B3HW[0].permute(1, 2, 0).mul(255).cpu().numpy()
    output_img = PImage.fromarray(output_img.astype(np.uint8))
    output_img.save(inpaint_path)
    print(f"Inpainted image saved as {inpaint_path}")

    orig_img = input_tensor[0].add(1).mul(0.5).permute(1, 2, 0).mul(255).cpu().numpy()
    orig_img = PImage.fromarray(orig_img.astype(np.uint8))
    
    comparison = Image.new('RGB', (256*3, 256))
    comparison.paste(orig_img, (0, 0))
    comparison.paste(masked_img, (256, 0))
    comparison.paste(output_img, (512, 0))
    comparison_path = os.path.join(output_dir, comparison_filename)
    comparison.save(comparison_path)
    print(f"Comparison image saved as {comparison_path} (original|masked|inpainted)")