"""Fig. 1: SDT-Net architecture (grayscale). Boxes are laid out on an explicit grid and checked
for overlaps; every arrow is declared as (source, target, style) so the drawn connections match
the information flow of the method exactly."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 7.2})
W, Hh = 100.0, 58.0
fig = plt.figure(figsize=(7.16, 4.15))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, Hh); ax.axis("off")

B = {}   # name -> (x, y, w, h)


def box(name, x, y, w, h, text, fill="white", lw=0.9, ls="-", bold=False):
    B[name] = (x, y, w, h)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fill, ec="black",
                                lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.0,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def side(name, s):
    x, y, w, h = B[name]
    f = 0.5
    if "@" in s:
        s, f = s.split("@"); f = float(f)
    return {"l": (x - 0.25, y + h * f), "r": (x + w + 0.25, y + h * f), "t": (x + w * f, y + h + 0.25),
            "b": (x + w * f, y - 0.25)}[s]


ARROWS = []


def arrow(a, sa, b, sb, ls="-", label=None, rad=0.0, lpos=0.5, loff=(0, 1.2), via=None):
    p, q = side(a, sa), side(b, sb)
    ARROWS.append((a, b, ls))
    if via is None:
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, ls=ls, color="black",
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
        mx, my = p[0] + (q[0] - p[0]) * lpos, p[1] + (q[1] - p[1]) * lpos
    else:  # orthogonal poly-line through way-points
        pts = [p] + via + [q]
        for u, v in zip(pts[:-2], pts[1:-1]):
            ax.plot([u[0], v[0]], [u[1], v[1]], color="black", lw=0.9, ls=ls)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=8, lw=0.9, ls=ls,
                                     color="black", shrinkA=0, shrinkB=0))
        mx, my = via[0]
    if label:
        ax.text(mx + loff[0], my + loff[1], label, ha="center", va="center", fontsize=6.4, style="italic",
                bbox=dict(fc="white", ec="none", pad=0.4))


# ---------------- stage bands ----------------
def band(x0, x1, title):
    ax.add_patch(FancyBboxPatch((x0, 1.0), x1 - x0, Hh - 2.0, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="#F4F4F4", ec="#9A9A9A", lw=0.6))
    ax.text((x0 + x1) / 2, Hh - 3.0, title, ha="center", va="center", fontsize=7.6, fontweight="bold")


band(0.8, 33.2, "(a) Anatomy-aware representation")
band(34.2, 67.2, "(b) Spectral stone twin")
band(68.2, 99.2, "(c) Calibrated decision layer")

# ---------------- (a) ----------------
box("unl", 2.5, 45.0, 13.0, 6.0, "Unlabelled and\ntraining volumes")
box("ssl", 18.0, 45.0, 13.5, 6.0, "Masked spectral\nreconstruction (SSL)")
box("inp", 2.5, 31.0, 13.0, 8.0, "Spectral CT  $\\mathbf{I}$\n$C$ channels\n+ protocol code $\\mathbf{m}$")
box("enc", 18.0, 31.0, 13.5, 8.0, "3D encoder\u2013decoder\n$f_{\\theta}$")
box("occ", 2.5, 19.5, 13.0, 6.0, "Occupancy logits\n$z(\\mathbf{x})$")
box("anat", 18.0, 19.5, 13.5, 6.0, "Anatomy head\n$\\mathbf{A}(\\mathbf{x})$")
box("gocc", 2.5, 5.5, 13.0, 7.0, "Gated occupancy\n$\\sigma(z+\\gamma\\log g)$", fill="#D9D9D9", bold=True)
box("gate", 18.0, 5.5, 13.5, 7.0, "Collecting-system\ngate $g(\\mathbf{x})$")

arrow("unl", "r", "ssl", "l")
arrow("ssl", "b", "enc", "t", label="initialise", loff=(4.6, 0), lpos=0.5)
arrow("inp", "r", "enc", "l")
arrow("enc", "b@0.75", "anat", "t@0.75")
arrow("enc", "b@0.25", "occ", "t", via=[(21.375, 28.0), (9.0, 28.0)])
arrow("anat", "b@0.75", "gate", "t@0.75", label="dilate", loff=(3.4, 0))
arrow("gate", "l", "gocc", "r")
arrow("occ", "b", "gocc", "t")

# ---------------- (b) ----------------
box("bgh", 36.0, 45.0, 14.0, 6.0, "Anatomy-rendered\ncounterfactual $\\hat{\\mathbf{B}}$")
box("exc", 52.5, 45.0, 13.0, 6.0, "Excess attenuation\n$\\mathbf{I}-\\hat{\\mathbf{B}}$")
box("inst", 36.0, 31.0, 14.0, 8.0, "Stone instances\n$\\Omega_k$ (support\nfrom PSF extent)")
box("mom", 52.5, 31.0, 13.0, 8.0, "Blur-invariant\nmoments $\\mathbf{M}_k$,\nmaterial angle $\\phi_k$")
box("par", 36.0, 19.5, 14.0, 6.0, "Twin parameters\n$\\boldsymbol{\\vartheta}_k=(\\mathbf{t},\\mathbf{a},\\mathbf{c})$")
box("ren", 52.5, 19.5, 13.0, 6.0, "PSF renderer\n$\\mathcal{R}(\\boldsymbol{\\vartheta}_k)$")
box("lik", 52.5, 5.5, 13.0, 7.0, "Noise-weighted\nlikelihood $\\mathcal{L}_k(D,\\hat{D})$")
box("lap", 36.0, 5.5, 14.0, 7.0, "Laplace covariance $\\boldsymbol{\\Sigma}_k$\n(Gauss\u2013Newton)", fill="#D9D9D9", bold=True)

arrow("anat", "r", "bgh", "l", via=[(33.4, 22.5), (33.4, 48.0)])
arrow("bgh", "r", "exc", "l")
arrow("gocc", "b", "inst", "l@0.25", via=[(9.0, 3.4), (34.7, 3.4), (34.7, 33.0)])
arrow("exc", "b", "mom", "t")
arrow("inst", "r", "mom", "l")
arrow("mom", "b", "ren", "t", label="initialise", loff=(4.4, 0))
arrow("par", "r", "ren", "l")
arrow("ren", "b", "lik", "t")
arrow("lik", "l@0.3", "lap", "r@0.3")
# reverse-mode gradient: from the likelihood back to the twin parameters (dashed, opposite to the forward path)
arrow("lik", "l@0.8", "par", "b@0.786", ls="--", via=[(51.2, 11.1), (51.2, 15.9), (47.0, 15.9)])
ax.text(46.2, 17.55, "reverse-mode\ngradient $\\nabla_{\\vartheta}\\mathcal{L}_k$", ha="right", va="center",
        fontsize=6.2, style="italic", linespacing=1.1)

# ---------------- (c) ----------------
box("feat", 70.5, 45.0, 13.0, 6.0, "Material + burden\nfeatures $\\boldsymbol{\\mu}_k$")
box("ens", 85.5, 45.0, 12.5, 6.0, "Propagated\nensemble head")
box("ood", 70.5, 31.0, 13.0, 8.0, "Material-manifold\nOOD score $d_k$")
box("conf", 85.5, 31.0, 12.5, 8.0, "Mondrian conformal\nsets $\\mathcal{C}_{\\alpha}$, $\\mathcal{C}^{\\mathrm{UA}}_{\\alpha}$\n(per spectral group)")
box("bur", 70.5, 19.5, 13.0, 6.0, "Fused burden\n$\\hat{V}_k$ with interval")
box("ref", 85.5, 19.5, 12.5, 6.0, "Referral rule\n$|\\mathcal{C}^{\\mathrm{UA}}_{\\alpha}|>1\\ \\vee\\ d_k>\\tau$")
box("out", 70.5, 5.5, 27.5, 7.0, "Patient-level report: stone map, composition\nset, volume interval, referral flag",
    fill="#D9D9D9", bold=True)

arrow("mom", "r@0.8", "feat", "l@0.75", via=[(67.0, 37.4), (67.0, 49.5)])
arrow("lap", "b", "feat", "l@0.25", via=[(43.0, 2.6), (69.2, 2.6), (69.2, 46.5)])
arrow("lap", "b", "bur", "l", via=[(43.0, 2.6), (69.2, 2.6), (69.2, 22.5)])
ax.plot([69.2], [22.5], "o", color="black", ms=2.2)
arrow("feat", "r", "ens", "l")
arrow("feat", "b", "ood", "t")
arrow("ens", "b", "conf", "t")
arrow("ood", "r", "ref", "l", via=[(84.6, 35.0), (84.6, 22.5)])
arrow("conf", "b", "ref", "t")
arrow("bur", "b", "out", "t@0.236")
arrow("ref", "b", "out", "t@0.7727")


def overlaps():
    names = list(B)
    bad = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x0, y0, w0, h0 = B[a]; x1, y1, w1, h1 = B[b]
            if x0 < x1 + w1 + 1.0 and x1 < x0 + w0 + 1.0 and y0 < y1 + h1 + 1.0 and y1 < y0 + h0 + 1.0:
                bad.append((a, b))
    return bad


if __name__ == "__main__":
    import sys
    print("overlapping boxes:", overlaps())
    print("arrows:", len(ARROWS))
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/figs/fig1_arch.png"
    fig.savefig(out, dpi=600)
