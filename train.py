"""Self-supervised spectral pretraining and supervised multi-task training of the backbone.

variants:  full (SSL + anatomy gate + background head), nossl, noanat, nobg,
           mtunet (plain multi-task U-Net), agunet (anatomy-gated U-Net, no SSL, no background head)
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
import data as Dt
from model import SDTBackbone, n_params

RUNS = os.environ.get("SDT_RUNS", "/home/claude/runs")
VARIANTS = {
    "full":   dict(ssl=True,  anat=True,  bg=True),
    "nossl":  dict(ssl=False, anat=True,  bg=True),
    "noanat": dict(ssl=True,  anat=False, bg=True),
    "nobg":   dict(ssl=True,  anat=True,  bg=False),
    "mtunet": dict(ssl=False, anat=False, bg=False),
    "agunet": dict(ssl=False, anat=True,  bg=False),
}
CROP = 32
ANAT_W = torch.tensor([1.0, 1.0, 3.0, 2.0, 1.0])


def crop_idx(rng, occ=None, p_stone=0.7):
    N = 48
    if occ is not None and rng.random() < p_stone and (occ > 0.3).any():
        v = np.argwhere(occ > 0.3)
        c = v[rng.integers(len(v))] + rng.integers(-10, 11, 3)
    else:
        c = rng.integers(CROP // 2, N - CROP // 2 + 1, 3)
    lo = np.clip(c - CROP // 2, 0, N - CROP)
    return tuple(slice(int(l), int(l) + CROP) for l in lo)


def flips(rng, *arrs):
    ax = [a for a in (-3, -2, -1) if rng.random() < 0.5]
    return [np.flip(a, ax).copy() if ax else a for a in arrs]


def ssl_pretrain(seed, iters=900, bs=4):
    fn = f"{RUNS}/ssl_s{seed}.pt"
    if os.path.exists(fn):
        return torch.load(fn)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X, MV = [], []
    for j in Dt.split_jobs("unl") + Dt.split_jobs("train"):    # image channels only (memory)
        xi = Dt.net_input(Dt.load_case(j)["img"], j["site"])
        X.append(xi[:2].astype(np.float16)); MV.append(xi[2:, 0, 0, 0].copy())
    X, MV = np.stack(X), np.stack(MV)
    m = SDTBackbone(True, True)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=iters, pct_start=0.1)
    t0 = time.time()
    for it in range(iters):
        xb, tb, mb = [], [], []
        for _ in range(bs):
            ii = rng.integers(len(X))
            x = X[ii][(slice(None),) + crop_idx(rng)].astype(np.float32)
            x = np.concatenate([x, np.broadcast_to(MV[ii].astype(np.float32)[:, None, None, None], (4,) + x.shape[1:])])
            (x,) = flips(rng, x)
            tgt = x[:2].copy()
            inp = x.copy()
            blk = rng.random((CROP // 4,) * 3) < 0.4
            mask = np.kron(blk, np.ones((4, 4, 4))).astype(bool)
            mk = np.stack([mask, mask])
            inp[:2][mk] = 0.0
            if inp[5, 0, 0, 0] > 0.5 and rng.random() < 0.5:       # spectral cross-prediction
                c = rng.integers(2)
                inp[c] = 0.0
                mk[c] = True
            xb.append(inp); tb.append(tgt); mb.append(mk)
        xb, tb, mb = map(lambda a: torch.tensor(np.stack(a)), (xb, tb, mb))
        rec = m(xb, ssl=True)
        loss = (torch.abs(rec - tb) * mb).sum() / mb.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it % 100 == 0:
            print(f"ssl {it} {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    sd = m.net.state_dict()
    torch.save(sd, fn)
    return sd


def train(variant, seed, iters=2000, bs=4, lam_a=0.5, lam_b=1.0):
    cfg = VARIANTS[variant]
    out = f"{RUNS}/{variant}_s{seed}.pt"
    if os.path.exists(out):
        return out
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = SDTBackbone(cfg["anat"], cfg["bg"])
    if cfg["ssl"]:
        m.net.load_state_dict(ssl_pretrain(seed))
    js, X, MV, Y = Dt.load_compact("train")
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=iters, pct_start=0.1)
    t0 = time.time()
    for it in range(iters):
        xb, ob, ab, bb, wb = [], [], [], [], []
        for _ in range(bs):
            i = rng.integers(len(X))
            sl = crop_idx(rng, Y[i]["occ"])
            x = Dt.expand(X[i][(slice(None),) + sl], MV[i])
            o = Y[i]["occ"][sl].astype(np.float32)
            a = Y[i]["anat"][sl].astype(np.int64)
            b = Y[i]["bg"][(slice(None),) + sl].astype(np.float32)
            w = 1.0 + 9.0 * ndi.binary_dilation(o > 0.01, iterations=3)
            x, o, a, b, w = flips(rng, x, o, a, b, w)
            xb.append(x); ob.append(o); ab.append(a); bb.append(b); wb.append(w)
        xb, ob, ab, bb, wb = [torch.tensor(np.stack(v)) for v in (xb, ob, ab, bb, wb)]
        r = m(xb)
        p = r["occ"]
        l_occ = F.binary_cross_entropy_with_logits(r["zs"], ob) + \
            (1 - (2 * (p * ob).sum() + 1) / (p.sum() + ob.sum() + 1))
        loss = l_occ
        if cfg["anat"]:
            loss = loss + lam_a * F.cross_entropy(r["anat"], ab, weight=ANAT_W)
        if cfg["bg"]:
            loss = loss + lam_b * ((r["bg"] - bb).abs().mean(1) * wb).sum() / wb.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step(); sch.step()
        if it % 100 == 0:
            print(f"{variant} s{seed} {it} {loss.item():.4f} occ {l_occ.item():.4f} {time.time()-t0:.0f}s", flush=True)
    torch.save(m.state_dict(), out)
    json.dump(dict(variant=variant, seed=seed, iters=iters, bs=bs, params=n_params(m), train_s=time.time() - t0),
              open(out + ".json", "w"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant"); ap.add_argument("seed", type=int)
    ap.add_argument("--iters", type=int, default=2000)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("NT", "2")))
    os.makedirs(RUNS, exist_ok=True)
    train(a.variant, a.seed, a.iters)
