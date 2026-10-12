"""Result figures (grayscale, no overlapping legends). Reads the JSON files written by run_eval.py."""
import os, sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = os.environ.get("SDT_RES", "/home/claude/results")
F = os.environ.get("SDT_FIG", "/home/claude/figs")
plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7.5,
                     "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
J = lambda n: json.load(open(f"{R}/{n}.json"))
METH = ["HU-LDA", "HU-RF", "MT-UNet", "AG-UNet", "SDT-Net"]
FILL = {"HU-LDA": "white", "HU-RF": "0.85", "MT-UNet": "0.65", "AG-UNet": "0.4", "SDT-Net": "0.1"}
HATCH = {"HU-LDA": "////", "HU-RF": "", "MT-UNet": "....", "AG-UNet": "", "SDT-Net": ""}
COH = [("test", "Internal test"), ("extD", "External D (DE)"), ("extE", "External E (SE)")]


def label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="bottom")


def fig_composition():
    C = J("comp_main")
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2))
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.27, top=0.9, wspace=0.42)
    for ax, (met, yl, lab) in zip(axs, [("macro_f1", "Macro-$F_1$", "(a)"), ("ece", "ECE", "(b)"),
                                        ("set_g2", "Mean set size (DE stones)", "(c)")]):
        w = 0.15
        for i, m in enumerate(METH):
            if m not in C:
                continue
            xs, ys, es = [], [], []
            for j, (sp, _) in enumerate(COH):
                if met in C[m][sp]:
                    xs.append(j + (i - 2) * w); ys.append(C[m][sp][met][0]); es.append(C[m][sp][met][1])
            ax.bar(xs, ys, w * 0.92, color=FILL[m], edgecolor="black", lw=0.5, hatch=HATCH[m], label=m,
                   yerr=es, error_kw=dict(lw=0.6, capsize=1.5))
        ax.set_xticks(range(3)); ax.set_xticklabels([c[1].replace(" (", "\n(") for c in COH], fontsize=6.6)
        ax.set_ylabel(yl); label(ax, lab)
        ax.grid(axis="y", alpha=0.3, lw=0.4); ax.set_axisbelow(True)
    axs[2].set_xlim(-0.5, 1.5); axs[2].set_xticks([0, 1]); axs[2].set_xticklabels(["Internal test\n(DE sites)", "External D\n(DE)"], fontsize=6.6)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.53, -0.01),
               handlelength=1.6, columnspacing=1.6)
    fig.savefig(f"{F}/fig_comp.png", dpi=600)


def fig_burden():
    S = J("size_bins")
    fig = plt.figure(figsize=(7.16, 2.4))
    names = ["HU threshold", "MT-UNet seg", "SDT-Net fused"]
    for i, n in enumerate(names):
        ax = fig.add_axes([0.065 + i * 0.205, 0.17, 0.17, 0.72])
        a = np.array(S["_scatter"][n])
        lr = np.log2(a[:, 1] / a[:, 0])
        ax.scatter(a[:, 0], lr, s=3.5, color="black", alpha=0.55, lw=0)
        b, sd = lr.mean(), lr.std()
        for v, ls in [(b, "-"), (b + 1.96 * sd, "--"), (b - 1.96 * sd, "--")]:
            ax.axhline(v, color="0.35", lw=0.8, ls=ls)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xscale("log"); ax.set_ylim(-2.2, 2.2); ax.set_xlim(0.8, 3000)
        ax.set_title(n, fontsize=7.3, pad=3)
        ax.set_xlabel("True volume (mm$^{3}$)")
        if i == 0:
            ax.set_ylabel("$\\log_{2}$(estimate / truth)")
        else:
            ax.set_yticklabels([])
        ax.text(0.97, 0.04, f"bias {b:+.2f}\nLoA $\\pm${1.96 * sd:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.2, bbox=dict(fc="white", ec="none", pad=0.6))
        ax.grid(alpha=0.25, lw=0.4)
        if i == 0:
            label(ax, "(a)", x=-0.36)
    ax = fig.add_axes([0.725, 0.17, 0.265, 0.72])
    bins = ["0-3", "3-4", "4-6", "6-10", "10-99"]
    styles = {"HU threshold": ("o", "white", "-"), "MT-UNet seg": ("s", "0.6", "-"), "AG-UNet seg": ("^", "0.6", "--"),
              "SDT-Net seg": ("v", "0.35", ":"), "SDT-Net twin": ("D", "0.15", "--"), "SDT-Net fused": ("o", "black", "-")}
    xs = np.arange(len(bins))
    for n, (mk, fc, ls) in styles.items():
        ys = [S[n][f"test:{b}"][0] if f"test:{b}" in S[n] else np.nan for b in bins]
        ax.plot(xs, ys, marker=mk, mfc=fc, mec="black", color="black", lw=0.8, ls=ls, ms=3.6, label=n)
    ax.set_yscale("log"); ax.set_ylim(1, 3000); ax.set_xticks(xs); ax.set_xticklabels(["<3", "3–4", "4–6", "6–10", ">10"])
    ax.set_xlabel("Maximum diameter (mm)"); ax.set_ylabel("MdAPE (%)")
    ax.grid(alpha=0.25, lw=0.4, which="both")
    ax.legend(fontsize=5.4, frameon=True, framealpha=1, edgecolor="0.7", loc="upper center", handlelength=2.0, ncol=2,
              columnspacing=0.8, handletextpad=0.4)
    label(ax, "(b)", x=-0.27)
    fig.savefig(f"{F}/fig_burden.png", dpi=600)


MK = {"UA": ("o", "white"), "COM": ("s", "black"), "COD": ("s", "white"), "CAP": ("^", "black"), "STR": ("D", "0.6"),
      "CYS": ("v", "0.6"), "BRU": ("P", "0.35"), "XAN": ("X", "0.35")}


def fig_plane_ood():
    Pp = J("material_plane"); O = J("ood")
    fig = plt.figure(figsize=(7.16, 2.5))
    ax = fig.add_axes([0.065, 0.16, 0.36, 0.76])
    for c in ["UA", "STR", "CYS", "COM", "COD", "CAP", "XAN", "BRU"]:
        if c not in Pp:
            continue
        a = np.array(Pp[c], dtype=float)
        ax.scatter(a[:, 0], a[:, 1], marker=MK[c][0], s=9, facecolor=MK[c][1], edgecolor="black", lw=0.4,
                   label=c + (" (OOD)" if c in ("BRU", "XAN") else ""), alpha=0.85)
    ax.set_xlabel("Material angle $\\phi$ (rad)"); ax.set_ylabel("Log basis magnitude $\\rho$")
    ax.set_xlim(-0.6, 2.4)
    ax.legend(fontsize=6, ncol=1, frameon=True, framealpha=1, edgecolor="0.7", loc="center left",
              bbox_to_anchor=(1.01, 0.5), handletextpad=0.3, markerscale=1.3)
    ax.grid(alpha=0.25, lw=0.4); label(ax, "(a)", x=-0.15)
    ax2 = fig.add_axes([0.66, 0.16, 0.32, 0.76])
    meths = [("manifold", "Material manifold $d_k$"), ("deep_maha", "Feature Mahalanobis"), ("mi", "Ensemble MI"),
             ("msp", "Max. softmax"), ("entropy", "Entropy")]
    y = np.arange(len(meths))
    for k, (key, lab) in enumerate(meths):
        for j, (m, off, fc) in enumerate([("SDT-Net", -0.25, "black"), ("AG-UNet", 0.0, "0.55"), ("MT-UNet", 0.25, "white")]):
            if m in O and key in O[m]:
                ax2.barh(k + off, O[m][key][0], 0.23, xerr=O[m][key][1], color=fc, edgecolor="black", lw=0.5,
                         error_kw=dict(lw=0.6, capsize=1.3), label=m if k == 0 else None)
    ax2.axvline(0.5, color="0.4", lw=0.7, ls="--")
    ax2.set_yticks(y); ax2.set_yticklabels([m[1] for m in meths], fontsize=6.6); ax2.invert_yaxis()
    ax2.set_xlim(0.2, 1.0); ax2.set_xlabel("AUROC, OOD phases vs internal test")
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(fc=fc, ec="black", lw=0.5, label=m) for m, fc in
                        [("SDT-Net", "black"), ("AG-UNet", "0.55"), ("MT-UNet", "white")]],
               fontsize=6, frameon=True, framealpha=1, edgecolor="0.7", loc="lower right")
    ax2.grid(axis="x", alpha=0.25, lw=0.4); label(ax2, "(b)", x=-0.5)
    fig.savefig(f"{F}/fig_plane.png", dpi=600)


def fig_conformal():
    A = J("conformal_classes"); T = J("treatment_referral")
    fig = plt.figure(figsize=(7.16, 2.3))
    ax = fig.add_axes([0.065, 0.2, 0.26, 0.68])
    al = [0.05, 0.1, 0.2]
    for g, mk in [(1, "o"), (2, "s")]:
        for mode, fc, ls in [("mond", "black", "-"), ("pool", "white", "--")]:
            ys = [A[f"{a}:{mode}:test:g{g}"]["cov"] for a in al]
            ax.plot(al, ys, marker=mk, mfc=fc, mec="black", color="black", ls=ls, lw=0.8, ms=3.6,
                    label=f"{'SE' if g == 1 else 'DE'}, {'Mondrian' if mode == 'mond' else 'pooled'}")
    ax.plot(al, [1 - a for a in al], color="0.5", lw=0.7, ls=":", label="nominal $1-\\alpha$")
    ax.set_xlabel("Miscoverage level $\\alpha$"); ax.set_ylabel("Coverage, internal test")
    ax.set_xticks(al); ax.set_ylim(0.7, 1.02)
    ax.legend(fontsize=5.6, frameon=True, framealpha=1, edgecolor="0.7", loc="lower left")
    ax.grid(alpha=0.25, lw=0.4); label(ax, "(a)", x=-0.25)
    ax = fig.add_axes([0.40, 0.2, 0.26, 0.68])
    for g, mk in [(1, "o"), (2, "s")]:
        for mode, fc, ls in [("mond", "black", "-"), ("pool", "white", "--")]:
            ys = [A[f"{a}:{mode}:test:g{g}"]["set"] for a in al]
            ax.plot(al, ys, marker=mk, mfc=fc, mec="black", color="black", ls=ls, lw=0.8, ms=3.6)
    ax.set_xlabel("Miscoverage level $\\alpha$"); ax.set_ylabel("Mean set size (of 6)"); ax.set_xticks(al)
    ax.grid(alpha=0.25, lw=0.4); label(ax, "(b)", x=-0.22)
    ax = fig.add_axes([0.735, 0.2, 0.255, 0.68])
    cats = [("test", "Int."), ("extD", "Ext. D"), ("extE", "Ext. E"), ("ood:XAN", "XAN")]
    w = 0.16
    for i, m in enumerate(METH):
        if m not in T:
            continue
        ys = []
        for sp, _ in cats:
            ys.append(T[m][sp]["refer"][0] if sp in T[m] else np.nan)
        ax.bar(np.arange(len(cats)) + (i - 2) * w, ys, w * 0.92, color=FILL[m], edgecolor="black", lw=0.5,
               hatch=HATCH[m], label=m)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([c[1] for c in cats], fontsize=6.6)
    ax.set_ylabel("Referral rate (UA decision)"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5.4, frameon=True, framealpha=1, edgecolor="0.7", loc="upper left", ncol=1)
    ax.grid(axis="y", alpha=0.25, lw=0.4); label(ax, "(c)", x=-0.25)
    fig.savefig(f"{F}/fig_conformal.png", dpi=600)


def fig_sensitivity():
    Pz = J("psf"); TA = J("twin_ablation")
    fig = plt.figure(figsize=(3.5, 2.6))
    ax = fig.add_axes([0.16, 0.18, 0.80, 0.68])
    scales = [k for k in ["0.80", "1.00", "1.25"] if k in Pz]
    for sp, mk, fc in [("test", "o", "black"), ("extD", "s", "0.55"), ("extE", "^", "white")]:
        ys = [Pz[s][sp]["vol_mdape"] for s in scales]
        ax.plot([float(s) for s in scales], ys, marker=mk, mfc=fc, mec="black", color="black", lw=0.8, ms=4,
                label={"test": "Internal test", "extD": "External D", "extE": "External E"}[sp])
    ax.set_xlabel("PSF width assumed by the twin / true PSF width"); ax.set_ylabel("Fused volume MdAPE (%)")
    ax.set_xticks([0.8, 1.0, 1.25]); ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=6.3, frameon=True, framealpha=1, edgecolor="0.7", loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.16))
    fig.savefig(f"{F}/fig_psf.png", dpi=600)


if __name__ == "__main__":
    what = sys.argv[1:] or ["comp", "burden", "plane", "conf", "psf"]
    for w in what:
        dict(comp=fig_composition, burden=fig_burden, plane=fig_plane_ood, conf=fig_conformal, psf=fig_sensitivity)[w]()
        print("done", w)
