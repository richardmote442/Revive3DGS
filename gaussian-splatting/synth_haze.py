# Synthesize physically-based depth-correlated haze (Koschmieder) on a clean 3DGS scene:
#   I_hazy = t(d) * I_clean + (1 - t(d)) * A ,   t(d) = exp(-beta * depth_norm)
# depth comes from a CLEAN-trained 3DGS render; I_clean is the original clean image.
import os, sys, torch
from argparse import ArgumentParser
from torchvision.utils import save_image
sys.path.insert(0, ".")
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
parser.add_argument("--iteration", type=int, default=-1)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--beta", type=float, default=2.5)
parser.add_argument("--airlight", type=float, nargs=3, default=[0.82, 0.85, 0.90])
args = get_combined_args(parser)
ds = model.extract(args); pipe = pipeline.extract(args)
g = GaussianModel(ds.sh_degree)
scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
bg = torch.tensor([1,1,1] if ds.white_background else [0,0,0], dtype=torch.float32, device="cuda")
A = torch.tensor(args.airlight, device="cuda").view(3,1,1)
os.makedirs(args.out_dir, exist_ok=True)
cams = list(scene.getTrainCameras()) + list(scene.getTestCameras())
for c in cams:
    with torch.no_grad():
        pkg = render(c, g, pipe, bg)
        invd = pkg["depth"].clamp_min(1e-6)
    clean = c.original_image.cuda().clamp(0,1)
    depth = 1.0 / invd
    if depth.shape[-2:] != clean.shape[-2:]:
        depth = torch.nn.functional.interpolate(depth[None], size=clean.shape[-2:], mode="bilinear", align_corners=False)[0]
    dn = (depth - depth.amin()) / (depth.amax() - depth.amin() + 1e-6)
    t = torch.exp(-args.beta * dn)
    hazy = (t * clean + (1.0 - t) * A).clamp(0,1)
    nm = c.image_name if c.image_name.endswith(".png") else c.image_name + ".png"
    save_image(hazy, os.path.join(args.out_dir, nm))
print("wrote", len(cams), "hazy images to", args.out_dir)
