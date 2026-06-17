#
# VeilGaussians: per-view latent degradation field (transmission t_v + airlight a_v).
# Composites a LOW-RANK / LOW-FREQUENCY 2D veil onto the CLEAN 3DGS render:
#       I_deg = t_v * I_clean + (1 - t_v) * a_v          (Koschmieder algebra)
# but t_v, a_v are FREE low-rank 2D fields from a per-view latent -> small MLP -> low-res grid
# -> bilinear upsample. Crucially: NO depth, NO scattering coefficient beta (that boundary keeps
# this distinct from a physical-medium model). The clean Gaussians are the canonical scene; at
# eval we drop the veil (t=1, a=0) and the canonical render is the restored result.
#
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class VeilField(nn.Module):
    def __init__(self, num_views, latent_dim=32, grid=16, hidden=128, t_init_logit=4.0):
        super().__init__()
        self.num_views = num_views
        self.grid = int(grid)
        self.t_init_logit = float(t_init_logit)   # +4 => sigmoid~0.982 => veil starts ~identity
        self.z = nn.Embedding(num_views, latent_dim)
        nn.init.normal_(self.z.weight, std=1e-2)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, self.grid * self.grid * 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _grids(self, idx):
        dev = self.z.weight.device
        idx = torch.as_tensor(idx, device=dev, dtype=torch.long).view(1)
        out = self.net(self.z(idx)).view(4, self.grid, self.grid)
        t_logit = out[0:1] + self.t_init_logit     # (1,g,g)
        a_logit = out[1:4]                          # (3,g,g)
        return t_logit, a_logit

    def forward(self, idx, H, W):
        t_logit, a_logit = self._grids(idx)
        t = torch.sigmoid(F.interpolate(t_logit.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0])
        a = torch.sigmoid(F.interpolate(a_logit.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0])
        return t, a   # t:(1,H,W) in (0,1], a:(3,H,W) in (0,1)

    def composite(self, I_clean, idx):
        C, H, W = I_clean.shape
        t, a = self.forward(idx, H, W)
        I_deg = t * I_clean + (1.0 - t) * a
        return I_deg, t, a
