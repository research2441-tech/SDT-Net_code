"""Fig. 2: (a) energy-dependent linear attenuation of the eight crystalline phases and urine (NIST/Elam data,
typical packing); (b)-(e) one development examination: low- and high-energy channels, ground-truth stone-free
background and anatomy labels."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import physics as P, simulator as S, data as Dt
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7.5})
fig = plt.figure(figsize=(7.16, 2.75))
ax = fig.add_axes([0.075, 0.15, 0.43, 0.78])
styles = {"UA": ("-", "black", 1.4), "COM": ("-", "0.25", 1.0), "COD": ("--", "0.25", 1.0), "CAP": ("-", "0.55", 1.4),
          "STR": (":", "black", 1.2), "CYS": ("-.", "black", 1.0), "BRU": ((0, (5, 1, 1, 1, 1, 1)), "0.45", 1.0),
          "XAN": ((0, (1, 1)), "0.55", 1.3)}
E = P.E_KEV
for st, (ls, c, lw) in styles.items():
    mu = P.stone_mu([st], [1.0], S.PACK[st][0])
    ax.semilogy(E, mu, ls=ls, color=c, lw=lw, label=st + (" (OOD)" if st in P.OOD_CLASSES else ""))
ax.semilogy(E, P.mu_spectrum("urine"), color="black", lw=0.8, ls=(0, (2, 2)), label="urine")
ax.set_xlim(20, 150); ax.set_ylim(0.15, 30)
ax.set_xlabel("Photon energy (keV)"); ax.set_ylabel("Linear attenuation (cm$^{-1}$)")
ax.legend(fontsize=6.0, frameon=False, ncol=2, loc="upper right", handlelength=2.6, columnspacing=0.8)
ax.text(-0.13, 1.0, "(a)", transform=ax.transAxes, fontweight="bold")
ax.grid(alpha=0.25, lw=0.4)
# example case
J = [j for j in Dt.jobs() if j["split"] == "train" and j["site"] == "A"]
best = None
for j in J[:200]:
    m = json.loads(str(np.load(Dt.case_path(j))["meta"]))
    if len(m["stones"]) >= 2 and m["n_calc"] > 0 and max(s["dmax"] for s in m["stones"]) > 6:
        best = j; break
c = Dt.load_case(best)
occ = c["occ"].astype(np.float32)
zc = int(np.argmax(occ.sum((1, 2))))
panels = [(c["img"][0].astype(np.float32)[zc], "(b) 80 kVp channel", (-150, 600)),
          (c["img"][1].astype(np.float32)[zc], "(c) Sn140 kVp channel", (-150, 600)),
          (c["bg"][0].astype(np.float32)[zc], "(d) Stone-free background", (-150, 600)),
          (c["anat"][zc].astype(np.float32), "(e) Anatomy and stones", None)]
pos = [(0.60, 0.545), (0.80, 0.545), (0.60, 0.085), (0.80, 0.085)]
for i, (im, title, win) in enumerate(panels):
    a = fig.add_axes([pos[i][0], pos[i][1], 0.165, 0.365])
    if win:
        a.imshow(im, cmap="gray", vmin=win[0], vmax=win[1], interpolation="nearest")
    else:
        lab = np.choose(im.astype(int), [0.0, 0.42, 0.82, 0.2, 0.2])
        lab[occ[zc] > 0.5] = 1.0
        a.imshow(lab, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    a.set_title(title, fontsize=7.0, pad=3)
a = fig.axes[-1]
a.text(0.5, -0.04, "white: stone; light grey:\ncollecting system", transform=a.transAxes, ha="center", va="top", fontsize=6.0)
fig.axes[3].text(0.5, -0.04, "window $-150$ to 600 HU", transform=fig.axes[3].transAxes, ha="center", va="top", fontsize=6.0)
fig.savefig("/home/claude/figs/fig2_data.png", dpi=600)
print(best, c["meta"]["stones"][0]["cls"], [s["cls"] for s in c["meta"]["stones"]])
