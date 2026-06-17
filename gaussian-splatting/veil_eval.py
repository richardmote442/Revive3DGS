#
# Evaluate a (vanilla or veil) 3DGS model: canonical render vs CLEAN gt/, mapped by sorted name order.
# If --veil_ckpt is given, also dump triptychs: [degraded input | composited veil render | canonical clean]
# plus the learned transmission t_v and airlight a_v maps.
#
import os, sys, json
import numpy as np
import torch
import torchvision.transforms.functional as tf
from torchvision.utils import save_image
from PIL import Image
from argparse import ArgumentParser
sys.path.insert(0, ".")
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from veil_field import VeilField

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", type=int, default=-1)
parser.add_argument("--gt_dir", required=True)
parser.add_argument("--veil_ckpt", default=None)
parser.add_argument("--triptych_out", default=None)
parser.add_argument("--n_triptych", type=int, default=3)
parser.add_argument("--tag", default="model")
args = get_combined_args(parser)

dataset = model.extract(args)
pipe = pipeline.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
bg = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], dtype=torch.float32, device="cuda")

train_cams = list(scene.getTrainCameras()); test_cams = list(scene.getTestCameras())
all_cams = sorted(train_cams + test_cams, key=lambda c: c.image_name)
gt_files = sorted(os.listdir(args.gt_dir))
assert len(gt_files) == len(all_cams), f"gt {len(gt_files)} != cams {len(all_cams)}"
name2gt = {c.image_name: gt_files[i] for i, c in enumerate(all_cams)}

def load_gt(fn, H, W):
    t = tf.to_tensor(Image.open(os.path.join(args.gt_dir, fn)).convert("RGB")).cuda()
    if t.shape[-2:] != (H, W):
        t = torch.nn.functional.interpolate(t[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    return t.clamp(0, 1)

@torch.no_grad()
def canon(c):
    return torch.clamp(render(c, gaussians, pipe, bg)["render"], 0, 1)

def evalset(cams):
    ps, ss, ls = [], [], []
    for c in cams:
        img = canon(c); gt = load_gt(name2gt[c.image_name], img.shape[1], img.shape[2])
        ps.append(psnr(img, gt).mean().item()); ss.append(ssim(img, gt).item())
        with torch.no_grad():
            ls.append(lpips(img.unsqueeze(0), gt.unsqueeze(0), net_type="vgg").item())
    return float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls)), len(cams)

res = {}
for nm, cams in [("train", train_cams), ("test", test_cams), ("all", all_cams)]:
    if not cams: continue
    p, s, l, n = evalset(cams)
    res[nm] = {"PSNR": p, "SSIM": s, "LPIPS": l, "n": n}
    print(f"[{args.tag}] [{nm}] canonical vs CLEAN gt: PSNR {p:.3f}  SSIM {s:.4f}  LPIPS {l:.4f}  ({n} views)")
print("JSON " + json.dumps({"tag": args.tag, "iteration": scene.loaded_iter, **res}))

if getattr(args,"veil_ckpt",None) and getattr(args,"triptych_out",None):
    ck = torch.load(args.veil_ckpt, map_location="cuda")
    veil = VeilField(len(ck["name2idx"]), latent_dim=ck["latent"], grid=ck["grid"]).cuda()
    veil.load_state_dict(ck["state_dict"]); veil.eval()
    os.makedirs(args.triptych_out, exist_ok=True)
    for c in sorted(train_cams, key=lambda c: c.image_name)[:args.n_triptych]:
        with torch.no_grad():
            clean = canon(c)
            idx = ck["name2idx"][c.image_name]
            deg, t, a = veil.composite(clean, idx)
        degraded_in = c.original_image.cuda().clamp(0, 1)
        row = torch.cat([degraded_in, deg.clamp(0,1), clean], dim=2)   # [input | composited | canonical]
        maps = torch.cat([t.repeat(3,1,1), a], dim=2)                  # [t (gray) | airlight a]
        nm = c.image_name.replace("/", "_")
        save_image(row, os.path.join(args.triptych_out, f"triptych_{nm}.png"))
        save_image(maps, os.path.join(args.triptych_out, f"veilmaps_{nm}.png"))
        print(f"  triptych saved: {nm}  (t mean={t.mean().item():.3f} min={t.min().item():.3f})")
    print(f"triptychs -> {args.triptych_out}")
