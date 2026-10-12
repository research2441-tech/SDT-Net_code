"""Fig.: one internal-test examination through SDT-Net (seed 1): input, ungated and gated occupancy,
anatomy-rendered counterfactual, excess attenuation, twin rendering and residual on the slice through a stone."""
import json, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data as Dt, twin as T
from model import SDTBackbone
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7.5})
torch.set_num_threads(1)
m = SDTBackbone(True, True); m.load_state_dict(torch.load("/home/claude/runs/full_s1.pt")); m.eval()
pick = None
for j in Dt.split_jobs("test"):
    if j["scan"] or j["site"] != "A":
        continue
    c = Dt.load_case(j)
    st = c["meta"]["stones"]
    if len(st) >= 1 and c["distract"].sum() > 3 and max(s["dmax"] for s in st) > 5:
        x = torch.tensor(Dt.net_input(c["img"], j["site"]))[None]
        with torch.no_grad():
            r = m(x)
        zs = torch.sigmoid(m.stone(r["feat"])[0, 0]).detach().numpy()
        # want a distractor that the ungated logit fires on
        hit = (zs > 0.5) & (c["distract"] > 0)
        zz = [z for z in range(48) if hit[z].any() and (c["occ"][z].astype(np.float32) > 0.5).sum() > 8]
        if zz:
            pick = (j, c, r, zs, zz[0]); break
j, c, r, zs, zdis = pick
site = j["site"]; img = c["img"].astype(np.float32)
occ = r["occ"][0].numpy(); pa = torch.softmax(r["anat"][0], 0).numpy()
bg = T.anat_background(img, pa, occ, site)
res = T.twin_fit(img, bg, occ, None, site, c["meta"]["wed"], 2)
q = max(res, key=lambda q: (q["mask_k"][zdis].sum(), q["V"]))
# render the fitted twin of the largest stone into the volume for display
K = q["mask_k"]
zc = zdis
D = img - bg
# re-render: build the same box rendering using twin internals
lo = np.maximum(np.argwhere(K).min(0) - 4, 0); hi = np.minimum(np.argwhere(K).max(0) + 5, 48)
sub = torch.stack(torch.meshgrid(*[torch.arange(l, h, dtype=torch.float32) for l, h in zip(lo, hi)], indexing="ij"), -1)
cov = np.diag(1.0 / q["a"] ** 2)
var_psf = np.array(T.calibrate_psf(site)); sig = np.sqrt(np.maximum(var_psf - 1 / 12, 0.05))
# orientation from the moment step (recomputed)
mu = np.array(q["centroid"])
Rt = torch.tensor(q["R"], dtype=torch.float32)
a = torch.tensor(q["a"], dtype=torch.float32)
rr = torch.sqrt((((sub - torch.tensor(mu, dtype=torch.float32)) @ Rt / a) ** 2).sum(-1) + 1e-9)
o = torch.sigmoid((1 - rr) * a.mean() / 0.16)
h = T.blur3(o, sig).numpy()
ren = np.zeros_like(D)
for e in range(2):
    ren[e][tuple(slice(l, hh) for l, hh in zip(lo, hi))] = q["c"][e] * h
P = [(img[0][zc], "(a) Input, 80 kVp", "gray", (-150, 600)),
     (zs[zc], "(b) Ungated occupancy", "gray", (0, 1)),
     (occ[zc], "(c) Gated occupancy $p$", "gray", (0, 1)),
     (bg[0][zc], "(d) Counterfactual $\\hat{B}$", "gray", (-150, 600)),
     (D[0][zc], "(e) Excess $D$", "gray", (-100, 1200)),
     (ren[0][zc], "(f) Twin rendering $\\hat{D}$", "gray", (-100, 1200)),
     (D[0][zc] - ren[0][zc], "(g) Residual $D-\\hat{D}$", "gray", (-300, 300))]
fig = plt.figure(figsize=(7.16, 1.55))
w, gap = 0.128, 0.0105
for i, (im, t, cm, win) in enumerate(P):
    ax = fig.add_axes([0.008 + i * (w + gap), 0.05, w, 0.76])
    ax.imshow(im, cmap=cm, vmin=win[0], vmax=win[1], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=6.8, pad=2.5)
fig.savefig("/home/claude/figs/fig_case.png", dpi=600)
s = c["meta"]["stones"]
print(j, [(x["cls"], round(x["vol"], 1)) for x in s], "fitted V", round(q["V"], 1), "ang", round(q["ang"], 2))
json.dump(dict(pid=j["pid"], stones=[(x["cls"], x["vol"], x["dmax"]) for x in s], V=q["V"], ang=q["ang"], chi=q["chi"]),
          open("/home/claude/results/case.json", "w"))
