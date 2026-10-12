"""Volumetric encoder-decoder with anatomy-constrained occupancy gating and a
counterfactual (stone-free) background head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ANAT = 5          # fat/other, parenchyma, collecting system, vessel, muscle
CS = 2              # collecting-system label index


def block(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1, bias=False), nn.GroupNorm(4, co), nn.LeakyReLU(0.1, True),
                         nn.Conv3d(co, co, 3, padding=1, bias=False), nn.GroupNorm(4, co), nn.LeakyReLU(0.1, True))


class UNet3D(nn.Module):
    def __init__(self, cin=6, w=(8, 16, 32, 64), drop=0.1):
        super().__init__()
        self.e1, self.e2, self.e3, self.b = block(cin, w[0]), block(w[0], w[1]), block(w[1], w[2]), block(w[2], w[3])
        self.u3, self.d3 = nn.ConvTranspose3d(w[3], w[2], 2, 2), block(2 * w[2], w[2])
        self.u2, self.d2 = nn.ConvTranspose3d(w[2], w[1], 2, 2), block(2 * w[1], w[1])
        self.u1, self.d1 = nn.ConvTranspose3d(w[1], w[0], 2, 2), block(2 * w[0], w[0])
        self.drop = nn.Dropout3d(drop)
        self.width = w

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool3d(e1, 2))
        e3 = self.e3(F.max_pool3d(e2, 2))
        b = self.drop(self.b(F.max_pool3d(e3, 2)))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return d1, b


class SDTBackbone(nn.Module):
    """use_anat: anatomy head + gate; use_bg: counterfactual background head."""

    def __init__(self, use_anat=True, use_bg=True, w=(8, 16, 32, 64)):
        super().__init__()
        self.net = UNet3D(6, w)
        c = w[0]
        self.use_anat, self.use_bg = use_anat, use_bg
        self.stone = nn.Conv3d(c, 1, 1)
        self.anat = nn.Conv3d(c, N_ANAT, 1) if use_anat else None
        self.bg = nn.Conv3d(c, 2, 1) if use_bg else None
        self.rec = nn.Conv3d(c, 2, 1)                       # self-supervised reconstruction head
        self.gamma = nn.Parameter(torch.tensor(1.0))        # gate strength

    def forward(self, x, ssl=False):
        f, b = self.net(x)
        if ssl:
            return self.rec(f)
        out = {"feat": f}
        z = self.stone(f)[:, 0]
        if self.use_anat:
            a = self.anat(f)
            out["anat"] = a
            pcs = torch.softmax(a, 1)[:, CS]
            g = F.max_pool3d(pcs[:, None], 3, 1, 1)[:, 0]    # dilated collecting-system support
            out["gate"] = g
            z = z + F.softplus(self.gamma) * torch.log(g.clamp_min(1e-4))
        out["zs"] = z
        out["occ"] = torch.sigmoid(z)
        if self.use_bg:
            out["bg"] = x[:, :2] + self.bg(f)               # residual parameterisation of stone-free HU/400
        return out


def n_params(m):
    return sum(p.numel() for p in m.parameters())
