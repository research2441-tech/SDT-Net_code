"""Patient-specific renal phantom + spectral CT acquisition simulator (RenalStone-DT-Sim).

Each examination is a 48 x 48 x 48 mm renal-sinus-centred region of interest generated on a
0.5 mm grid, rendered for every spectral channel of the site protocol, blurred by the in-plane
point-spread function and the slice-sensitivity profile, sampled at the slice positions,
corrupted by correlated noise and interpolated to the 1.0 mm analysis grid. A stone-free
rendering of the same anatomy is produced as the counterfactual background target.
"""
import numpy as np
from scipy import ndimage as ndi
import physics as P

HR = 0.5          # high-resolution phantom spacing (mm)
NH = 96           # high-res voxels per axis (48 mm)
NG = 48           # analysis grid voxels per axis (1 mm)
LAB = dict(fat=0, parenchyma=1, collecting=2, vessel=3, muscle=4)

MIXES = {"COM": ["COD", "CAP", "UA"], "COD": ["COM", "CAP"], "CAP": ["COM", "STR"],
         "UA": ["COM"], "STR": ["CAP"], "CYS": ["COM"]}
PACK = {"UA": (0.85, 0.05), "COM": (0.80, 0.06), "COD": (0.68, 0.07), "CAP": (0.42, 0.09),
        "STR": (0.74, 0.07), "CYS": (0.84, 0.05), "BRU": (0.78, 0.06), "XAN": (0.82, 0.06)}
PRIOR = {"COM": .36, "COD": .14, "CAP": .14, "UA": .17, "STR": .10, "CYS": .09}

_c = (np.arange(NH) + 0.5) * HR - 24.0
ZZ, YY, XX = np.meshgrid(_c, _c, _c, indexing="ij")
PTS = np.stack([XX, YY, ZZ], -1)


def rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def ellipsoid(c, ax, R):
    d = (PTS - c) @ R
    return ((d / ax) ** 2).sum(-1)


def tube(p0, p1, r):
    v = p1 - p0
    L2 = (v ** 2).sum()
    t = np.clip(((PTS - p0) @ v) / L2, 0, 1)
    q = p0 + t[..., None] * v
    return np.sqrt(((PTS - q) ** 2).sum(-1)) <= r


def smooth_field(rng, amp, sigma=6.0):
    f = ndi.gaussian_filter(rng.normal(size=(NH, NH, NH)).astype(np.float32), sigma / HR)
    return amp * f / (f.std() + 1e-6)


def make_anatomy(rng):
    lab = np.zeros((NH, NH, NH), np.uint8)
    p0 = rng.uniform(-3, 3, 3)
    hydro = rng.random() < 0.2
    sc = rng.uniform(1.3, 1.7) if hydro else 1.0
    # psoas / muscle slab medial-posterior
    mus = (XX < -15 + rng.uniform(-2, 2)) & (YY > 6 + rng.uniform(-2, 2))
    lab[mus] = LAB["muscle"]
    kid = ellipsoid(p0 + np.array([9, 0, 0]), np.array([27, 21, 30]) * rng.uniform(.9, 1.1, 3), rot(rng) * 0 + np.eye(3)) <= 1
    sinus = ellipsoid(p0 + np.array([-5, 0, 0]), np.array([17, 10, 15]) * rng.uniform(.9, 1.1, 3), np.eye(3)) <= 1
    lab[kid] = LAB["parenchyma"]
    lab[kid & sinus] = LAB["fat"]
    # vessels from hilum to medial edge
    hil = p0 + np.array([-8, 0, 0])
    art_r, vein_r = rng.uniform(2.0, 3.0), rng.uniform(3.0, 4.2)
    a0, a1 = hil + np.array([0, -3, 3]), np.array([-24.5, rng.uniform(-6, 2), p0[2] + rng.uniform(2, 8)])
    v0, v1 = hil + np.array([0, 3, -1]), np.array([-24.5, rng.uniform(0, 8), p0[2] + rng.uniform(-6, 2)])
    art = tube(a0, a1, art_r)
    lab[art | tube(v0, v1, vein_r)] = LAB["vessel"]
    # collecting system: pelvis + infundibula/calyces + ureter
    Rp = rot(rng)
    pel = ellipsoid(p0, np.array([rng.uniform(7, 10), rng.uniform(4.5, 6.5), rng.uniform(4.5, 6.5)]) * sc, Rp) <= 1
    cs = pel.copy()
    cups = []
    nc = rng.integers(3, 6)
    for k in range(nc):
        th = 2 * np.pi * (k / nc) + rng.uniform(-.4, .4)
        d = np.array([0.75 + rng.uniform(0, .4), 0.55 * np.cos(th), 1.2 * np.sin(th)])
        d /= np.linalg.norm(d)
        L = rng.uniform(11, 16)
        tip = p0 + d * L
        cs |= tube(p0, tip, rng.uniform(1.3, 2.1) * sc)
        rc = rng.uniform(2.6, 3.8) * sc
        cs |= ((PTS - tip) ** 2).sum(-1) <= rc ** 2
        cups.append((tip, rc))
    ure_end = np.array([p0[0] - rng.uniform(4, 9), p0[1] + rng.uniform(-3, 3), -24.5])
    upj = p0 + (ure_end - p0) * 0.35
    cs |= tube(p0, ure_end, rng.uniform(1.6, 2.4) * (1.2 if hydro else 1.0))
    lab[cs] = LAB["collecting"]
    return dict(lab=lab, p0=p0, cups=cups, upj=upj, ure=(p0, ure_end), art=(a0, a1, art_r), hydro=hydro)


def stone_shape(rng, c, deq):
    ax = np.array([1.0, rng.uniform(.65, 1.0), rng.uniform(.6, 1.0)])
    ax = ax * (deq / 2) / np.prod(ax) ** (1 / 3)
    R = rot(rng)
    d = (PTS - c) @ R
    r = np.sqrt(((d / ax) ** 2).sum(-1)) + 1e-9
    u = d / np.linalg.norm(d, axis=-1, keepdims=True).clip(1e-9)
    pert = np.zeros_like(r)
    rough = rng.uniform(0.03, 0.12)
    for _ in range(4):
        w = rng.normal(size=3) * rng.uniform(1.5, 3.5)
        pert += np.cos(u @ w + rng.uniform(0, 2 * np.pi))
    rr = r / (1 + rough * pert / 2)
    return rr, ax


def place_stones(rng, A, n, ood=False):
    stones = []
    occ_all = np.zeros((NH, NH, NH), np.float32)
    for k in range(n):
        u = rng.random()
        if u < 0.68:
            tip, rc = A["cups"][rng.integers(len(A["cups"]))]
            loc, c, lim = "calyx", tip + rng.normal(0, .6, 3), 2 * rc + 3
        elif u < 0.90:
            loc, c, lim = "pelvis", A["p0"] + rng.normal(0, 2.0, 3), 16
        else:
            loc, c, lim = "upj", A["upj"] + rng.normal(0, .6, 3), 9
        deq = float(np.clip(np.exp(rng.normal(np.log(4.2), 0.55)), 1.5, min(lim, 15)))
        if ood:
            main = OODpick(rng)
        else:
            main = rng.choice(list(PRIOR), p=np.array(list(PRIOR.values())))
        mixed = rng.random() < 0.35 and not ood
        if mixed:
            sec = rng.choice(MIXES[main])
            fsec = rng.uniform(0.10, 0.45)
            mode = "core" if rng.random() < 0.5 else "homog"
        else:
            sec, fsec, mode = None, 0.0, "pure"
        mpk, spk = PACK[main]
        pack = float(np.clip(rng.normal(mpk, spk), 0.25, 0.97))
        rr, ax = stone_shape(rng, c, deq)
        m = rr <= 1.0
        if m.sum() < 2:
            continue
        stones.append(dict(cls=main, sec=sec, fsec=fsec, mode=mode, pack=pack, loc=loc, c=c,
                           rr=rr, mask=m, ood=bool(ood), lam=rng.uniform(0.8, 1.6),
                           lam_amp=rng.uniform(0.0, 0.08) if main in ("COM", "CAP", "STR") else 0.0))
    return stones


def OODpick(rng):
    return P.OOD_CLASSES[rng.integers(2)]


def hu_tables(site, wed):
    ws = P.protocol_weights(site, wed)
    tis = {k: np.array([P.hu_of(P.mu_spectrum(k), w) for w in ws]) for k in P.TISSUES}
    ph = {k: np.array([P.hu_of(P.mu_spectrum(k), w) for w in ws]) for k in P.STONES}
    return ws, tis, ph


def build_hu(A, stones, calc, site, wed, rng, fields, with_stones=True):
    ws, tis, ph = hu_tables(site, wed)
    C = len(ws)
    lab = A["lab"]
    base = {LAB["fat"]: "adipose", LAB["parenchyma"]: "kidney", LAB["collecting"]: "urine",
            LAB["vessel"]: "blood", LAB["muscle"]: "muscle"}
    out = np.zeros((C, NH, NH, NH), np.float32)
    for l, name in base.items():
        m = lab == l
        for c in range(C):
            out[c][m] = tis[name][c]
    tex = fields
    for c in range(C):
        out[c] += tex * (lab != LAB["collecting"])
    # calcified distractors (vascular wall plaques, parenchymal calcifications)
    for (m, pk) in calc:
        for c in range(C):
            hcal = pk * ph["CAP"][c] + (1 - pk) * tis["blood"][c]
            out[c][m] = hcal
    if with_stones:
        hu_u = tis["urine"]
        for s in stones:
            m = s["mask"]
            rr = s["rr"][m]
            pk = s["pack"] * (1 + s["lam_amp"] * np.sin(2 * np.pi * rr * 3.0 / s["lam"]))
            for c in range(C):
                hm = ph[s["cls"]][c]
                if s["sec"] is None:
                    solid = np.full(rr.shape, hm)
                else:
                    hs = ph[s["sec"]][c]
                    if s["mode"] == "homog":
                        solid = np.full(rr.shape, (1 - s["fsec"]) * hm + s["fsec"] * hs)
                    else:  # core of the secondary phase occupying fsec of the volume
                        rc = s["fsec"] ** (1 / 3)
                        solid = np.where(rr <= rc, hs, hm)
                out[c][m] = pk * solid + (1 - pk) * hu_u[c]
    return out


def calcifications(rng, A):
    calc = []
    if rng.random() < 0.5:
        a0, a1, ar = A["art"]
        for _ in range(rng.integers(1, 4)):
            t = rng.uniform(0.1, 0.9)
            q = a0 + t * (a1 - a0)
            n = np.cross(a1 - a0, rng.normal(size=3))
            n /= np.linalg.norm(n)
            cc = q + n * ar
            r = rng.uniform(0.6, 1.5)
            m = ellipsoid(cc, np.array([r * 1.6, r, r * 0.8]), rot(rng)) <= 1
            calc.append((m, rng.uniform(0.25, 0.6)))
    if rng.random() < 0.3:
        par = np.argwhere(A["lab"] == LAB["parenchyma"])
        for _ in range(rng.integers(1, 4)):
            v = par[rng.integers(len(par))]
            cc = PTS[v[0], v[1], v[2]]
            r = rng.uniform(0.5, 1.2)
            calc.append(((((PTS - cc) ** 2).sum(-1) <= r * r), rng.uniform(0.2, 0.5)))
    return calc


def acquire(hu, site, wed, rng, noise=True, phase=None):
    """blur, slice-sample, add noise, interpolate to the 1 mm grid."""
    pr = P.PROTOCOLS[site]
    sxy = pr["fwhm"] / 2.355 / HR
    s_mm = pr["slice"]
    C = hu.shape[0]
    if phase is None:
        phase = rng.uniform(0, s_mm)
    zc = np.arange(-24 + phase - s_mm, 24 + s_mm, s_mm)      # slice centres (mm)
    out = np.zeros((C, NG, NG, NG), np.float32)
    nscale = np.exp(0.035 * (wed - 28.0))
    for c in range(C):
        v = ndi.gaussian_filter(hu[c], (sxy, sxy, sxy), mode="nearest")
        w = max(int(round(s_mm / HR)), 1)
        v = ndi.uniform_filter1d(v, w, axis=0, mode="nearest") if w > 1 else v
        # in-plane: 2x2 area average to 1 mm
        v = v.reshape(NH, NG, 2, NG, 2).mean((2, 4))
        # sample slices at zc (linear in hr z index)
        zi = (zc + 24.0) / HR - 0.5
        sl = ndi.map_coordinates(v, np.stack(np.meshgrid(zi, np.arange(NG), np.arange(NG), indexing="ij")),
                                 order=1, mode="nearest")
        if noise:
            nz = rng.normal(size=sl.shape).astype(np.float32)
            nz = ndi.gaussian_filter(nz, (0, 0.7, 0.7))
            nz *= pr["noise"][c] * nscale / (nz.std() + 1e-6)
            sl = sl + nz
        # interpolate slices to 1 mm grid centres
        zg = (np.arange(NG) + 0.5) - 24.0
        zi2 = (zg - zc[0]) / s_mm
        out[c] = ndi.map_coordinates(sl, np.stack(np.meshgrid(zi2, np.arange(NG), np.arange(NG), indexing="ij")),
                                     order=1, mode="nearest")
    return out, phase


def down2(x):
    return x.reshape(NG, 2, NG, 2, NG, 2).mean((1, 3, 5))


def simulate_case(seed, site, n_stones=None, ood=False, wed=None, labelled=True, noise_seed=None):
    rng = np.random.default_rng(seed)
    if wed is None:
        mu = 30.0 if site == "D" else 27.0
        wed = float(np.clip(rng.normal(mu, 4.0), 18, 42))
    A = make_anatomy(rng)
    if n_stones is None:
        n_stones = rng.choice([0, 1, 2, 3, 4], p=[.15, .45, .25, .10, .05])
    stones = place_stones(rng, A, n_stones, ood=ood)
    # stones sit in urine: grow collecting system around them
    for s in stones:
        A["lab"][ndi.binary_dilation(s["mask"], iterations=1)] = LAB["collecting"]
    calc = calcifications(rng, A)
    fields = smooth_field(rng, 6.0)
    hu = build_hu(A, stones, calc, site, wed, rng, fields, True)
    ns = seed + 7_000_003 if noise_seed is None else noise_seed
    st = np.random.default_rng(ns)
    img, phase = acquire(hu, site, wed, st)
    out = dict(img=img.astype(np.float16), site=site, wed=wed, phase=phase)
    if not labelled:
        return out
    hu0 = build_hu(A, stones, calc, site, wed, rng, fields, False)
    bg, _ = acquire(hu0, site, wed, None, noise=False, phase=phase)
    clean, _ = acquire(hu, site, wed, None, noise=False, phase=phase)
    occ = np.zeros((NG, NG, NG), np.float32)
    inst = np.zeros((NG, NG, NG), np.uint8)
    best = np.zeros((NG, NG, NG), np.float32)
    recs = []
    for i, s in enumerate(stones):
        o = down2(s["mask"].astype(np.float32))
        occ = np.maximum(occ, o)
        sel = o > best
        inst[sel & (o > 0)] = i + 1
        best = np.maximum(best, o)
        vol = float(s["mask"].sum() * HR ** 3)
        pts = PTS[s["mask"]]
        ext = pts.max(0) - pts.min(0) + HR
        recs.append(dict(id=i + 1, cls=s["cls"], sec=s["sec"], fsec=s["fsec"], mode=s["mode"], pack=s["pack"],
                         loc=s["loc"], vol=vol, deq=float((6 * vol / np.pi) ** (1 / 3)), dmax=float(ext.max()),
                         c=[float(x) for x in s["c"]], ood=s["ood"]))
    lab = A["lab"]
    # anatomy label on analysis grid by majority (mode) of 2x2x2 blocks
    oh = np.stack([(lab == l) for l in range(5)]).astype(np.float32)
    anat = oh.reshape(5, NG, 2, NG, 2, NG, 2).mean((2, 4, 6)).argmax(0).astype(np.uint8)
    dis = np.zeros((NH, NH, NH), bool)
    for m, _ in calc:
        dis |= m
    out.update(bg=bg.astype(np.float16), clean=clean.astype(np.float16), occ=occ.astype(np.float16),
               inst=inst, anat=anat, distract=(down2(dis.astype(np.float32)) > 0.25).astype(np.uint8),
               stones=recs, hydro=A["hydro"], n_calc=len(calc))
    return out


if __name__ == "__main__":
    import time
    t = time.time()
    o = simulate_case(1, "A", n_stones=3)
    print(time.time() - t, o["img"].shape, [(s["cls"], round(s["deq"], 1), s["loc"]) for s in o["stones"]])
    print(float(o["img"].max()), float(o["bg"].max()))
