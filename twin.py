"""Spectral stone twin: blur-invariant excess-attenuation moments, differentiable
analysis-by-synthesis refinement of a parametric stone model, Laplace covariance and
protocol-invariant material coordinates."""
import functools
import numpy as np
import torch
from scipy import ndimage as ndi
import physics as P
import simulator as S

ETA = 4 * np.pi * 0.7 ** 2          # effective number of correlated voxels per independent noise sample


@functools.lru_cache(maxsize=None)
def calibrate_psf(site, n=24):
    """Effective point-response second moments (voxel^2) on the 1 mm analysis grid,
    obtained by passing sub-voxel point objects through the acquisition chain."""
    rng = np.random.default_rng(5)
    vs = []
    for _ in range(n):
        hu = np.zeros((1, S.NH, S.NH, S.NH), np.float32)
        i = rng.integers(44, 52, 3)
        hu[0, i[0], i[1], i[2]] = 1e6
        pr = dict(P.PROTOCOLS[site])
        out, _ = S.acquire(hu, site, 28.0, rng, noise=False)
        v = out[0].astype(np.float64)
        v = np.clip(v, 0, None)
        g = np.indices(v.shape).reshape(3, -1).astype(np.float64)
        w = v.reshape(-1) / v.sum()
        mu = (g * w).sum(1)
        cov = ((g - mu[:, None]) ** 2 * w).sum(1)
        vs.append(cov)
    return tuple(np.mean(vs, 0))          # (z, y, x) variances in voxel^2


@functools.lru_cache(maxsize=4096)
def basis(site, wed_r):
    """contrast basis B[e,m] = HU(m) - HU(urine) for channels e and bases m = (organic UA, apatite)."""
    ws = P.protocol_weights(site, float(wed_r))
    B = np.array([[P.hu_of(P.mu_spectrum(m), w) - P.hu_of(P.mu_spectrum("urine"), w) for m in ("UA", "CAP")]
                  for w in ws])
    hu_u = np.array([P.hu_of(P.mu_spectrum("urine"), w) for w in ws])
    return B, hu_u


def gauss1d(sig, r=4):
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / max(sig, 1e-3)) ** 2)
    return k / k.sum()


def blur3(v, sig):
    """separable Gaussian blur of a (D,H,W) tensor with per-axis sigmas (voxels)."""
    out = v[None, None]
    for ax, s in enumerate(sig):
        k = gauss1d(float(s))
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = len(k)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = len(k) // 2
        out = torch.nn.functional.conv3d(torch.nn.functional.pad(out, pad, mode="replicate"), k.view(shape))
    return out[0, 0]


def anat_background(img, pa, occ, site):
    """Anatomy-rendered counterfactual: soft anatomy probabilities times per-examination tissue attenuation
    (robust medians over confidently labelled, stone-free, eroded regions), blurred by the residual PSF."""
    C = img.shape[0]
    lab, conf = pa.argmax(0), pa.max(0)
    sig = np.sqrt(np.maximum(np.array(calibrate_psf(site)) - 1 / 12, 0.05))
    bg = np.zeros_like(img, dtype=np.float32)
    for e in range(C):
        tau = np.zeros(pa.shape[0])
        for l in range(pa.shape[0]):
            sel = (lab == l) & (conf > 0.9) & (occ < 0.02)
            if sel.sum() > 50:
                sel = ndi.binary_erosion(sel)
            if sel.sum() > 10:
                tau[l] = np.median(img[e][sel])
            elif (lab == l).any():
                tau[l] = np.median(img[e][lab == l])
        b = (pa * tau[:, None, None, None]).sum(0)
        bg[e] = blur3(torch.tensor(b, dtype=torch.float32), sig).numpy()
    return bg


def noise_sigma(img, bg, occ):
    """robust per-channel noise estimate from background residuals away from candidate stones."""
    m = occ < 0.02
    if m.sum() < 500:
        m = occ <= np.quantile(occ, 0.5)
    s = []
    for c in range(img.shape[0]):
        r = (img[c] - bg[c])[m]
        s.append(1.4826 * np.median(np.abs(r - np.median(r))) + 1e-3)
    return np.array(s)


def instances(occ, thr=0.5, peak=0.6):
    lab, n = ndi.label(occ > thr, structure=np.ones((3, 3, 3)))
    keep = []
    for k in range(1, n + 1):
        if occ[lab == k].max() >= peak:
            keep.append(k)
    if len(keep) > 12:                      # cap on candidate count per examination
        sz = {k: (lab == k).sum() for k in keep}
        keep = sorted(keep, key=lambda k: -sz[k])[:12]
    out = np.zeros_like(lab)
    for i, k in enumerate(keep):
        out[lab == k] = i + 1
    return out, len(keep)


def twin_fit(img, bg, occ, feat, site, wed, n_ch, refine=True, iters=30, bg_mode="net", psf_scale=1.0, occ_prior=0.8):
    """img/bg: (C,D,H,W) HU numpy; occ: (D,H,W); feat: (F,D,H,W) or None.
    Returns a list of per-instance dicts."""
    C = n_ch
    img = img[:C].astype(np.float64)
    var_psf = np.array(calibrate_psf(site)) * psf_scale ** 2
    sig_psf = np.sqrt(var_psf)
    sig_twin = np.sqrt(np.maximum(var_psf - 1 / 12, 0.05))
    lab, n = instances(occ)
    if n == 0:
        return []
    rz = int(np.ceil(3 * sig_psf[0])) + 1
    rxy = int(np.ceil(3 * max(sig_psf[1:]))) + 1
    st = np.zeros((2 * rz + 1, 2 * rxy + 1, 2 * rxy + 1), bool)
    zz, yy, xx = np.indices(st.shape)
    st = ((zz - rz) / rz) ** 2 + ((yy - rxy) / rxy) ** 2 + ((xx - rxy) / rxy) ** 2 <= 1.0001
    # nearest-instance ownership so supports of neighbouring stones do not double count
    dist, idx = ndi.distance_transform_edt(lab == 0, sampling=(1.0, 1.0, 1.0), return_indices=True)
    owner = lab[idx[0], idx[1], idx[2]]
    if bg_mode == "net":
        bgc = bg[:C].astype(np.float64)
    sig_n = None
    B, hu_u = basis(site, round(wed))
    res = []
    for k in range(1, n + 1):
        K = lab == k
        Om = ndi.binary_dilation(K, structure=st) & (owner == k)
        if bg_mode == "annulus":
            shell = Om & ~ndi.binary_dilation(K, iterations=2)
            bgc = np.zeros_like(img)
            for c in range(C):
                bgc[c] = np.median(img[c][shell]) if shell.any() else np.median(img[c])
        if sig_n is None or bg_mode == "annulus":
            sig_n = noise_sigma(img, bgc, occ)
        D = (img - bgc)[:, Om]                                   # (C, n_vox)
        pts = np.argwhere(Om).astype(np.float64)                 # (n, 3) z,y,x
        M = D.sum(1)                                             # HU*mm^3 per channel (blur invariant)
        wcomb = (D * (M / sig_n[:C] ** 2)[:, None]).sum(0)
        wsum = wcomb.sum()
        if wsum <= 0:
            continue
        mu = (pts * wcomb[:, None]).sum(0) / wsum
        dd = pts - mu
        cov_obs = (dd[:, :, None] * dd[:, None, :] * wcomb[:, None, None]).sum(0) / wsum
        cov_s = cov_obs - np.diag(var_psf)
        if not np.all(np.isfinite(cov_s)):
            continue
        lam, R = np.linalg.eigh(cov_s)
        lam = np.clip(lam, 0.15 ** 2, None)
        a0 = np.sqrt(5 * lam)
        if occ_prior:
            # occupancy-shape floor: the network occupancy is a (nearly) deblurred support, so its second moments
            # bound the semi-axes from below when the deconvolved intensity moments are ill-conditioned
            po = occ[Om]
            mo = (pts * po[:, None]).sum(0) / po.sum()
            do = pts - mo
            Co = (do[:, :, None] * do[:, None, :] * po[:, None, None]).sum(0) / po.sum()
            lam_o = np.einsum("ij,jk,ki->i", R.T, Co, R)
            a0 = np.maximum(a0, occ_prior * np.sqrt(5 * np.clip(lam_o, 0.15 ** 2, None)))
        V0 = 4 / 3 * np.pi * np.prod(a0)
        c0 = M / V0
        out = dict(k=k, M=M, mu=mu, a_mom=a0, V_mom=V0, c_mom=c0, n_vox=int(Om.sum()), sig_n=sig_n[:C].copy(),
                   V_seg=float(occ[Om].sum()), occ_max=float(occ[K].max()))
        # ---------- differentiable analysis-by-synthesis refinement ----------
        lo = np.maximum(pts.min(0).astype(int) - 1, 0)
        hi = np.minimum(pts.max(0).astype(int) + 2, np.array(occ.shape))
        box = tuple(slice(l, h) for l, h in zip(lo, hi))
        Dbox = torch.tensor((img - bgc)[(slice(None),) + box], dtype=torch.float32)
        Wbox = torch.tensor(Om[box], dtype=torch.float32)
        shp = Dbox.shape[1:]
        sub = torch.stack(torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shp],
                                         indexing="ij"), -1)
        Rt = torch.tensor(R, dtype=torch.float32)
        mu_t = torch.tensor(mu - lo, dtype=torch.float32)
        sigN = torch.tensor(sig_n[:C], dtype=torch.float32)
        prior_la = torch.tensor(np.log(a0), dtype=torch.float32)

        def render(theta):
            m_, la, cc = theta[:3], theta[3:6], theta[6:]
            a = torch.exp(la)
            r = torch.sqrt((((sub - m_) @ Rt / a) ** 2).sum(-1) + 1e-9)
            o = torch.sigmoid((1 - r) * a.mean() / 0.16)      # logistic edge ~ one-voxel box integration
            h = blur3(o, sig_twin)
            return cc[:, None, None, None] * 1000.0 * h[None]

        def loss_fn(theta):
            pred = render(theta)
            r = (Dbox - pred) / sigN[:, None, None, None]
            return 0.5 * (r ** 2 * Wbox).sum() / ETA + 0.5 * (((theta[3:6] - prior_la) / 0.5) ** 2).sum()

        theta = torch.tensor(np.concatenate([mu - lo, np.log(a0), c0 / 1000.0]), dtype=torch.float32,
                             requires_grad=True)
        if refine:
            opt = torch.optim.Adam([theta], lr=0.05)
            for it in range(iters):
                opt.zero_grad()
                L = loss_fn(theta)
                L.backward()                               # reverse-mode gradient through the renderer
                opt.step()
                with torch.no_grad():
                    theta[3:6].clamp_(np.log(0.15), np.log(12.0))
        th = theta.detach()
        with torch.no_grad():
            pred = render(th)
            r = ((Dbox - pred) / sigN[:, None, None, None]) ** 2 * Wbox
            nu = max(float(Wbox.sum()) * C / ETA - len(th), 1.0)
            chi = float(r.sum()) / ETA / nu
        # Laplace covariance from the Gauss-Newton approximation of the twin likelihood
        J = torch.func.jacfwd(lambda t: (render(t) / sigN[:, None, None, None])[:, Wbox > 0].reshape(-1))(th)
        Hm = (J.T @ J).double().numpy() / ETA
        Hm[3:6, 3:6] += np.eye(3) / 0.5 ** 2
        Hm += np.eye(len(th)) * 1e-6
        Cov = np.linalg.inv(Hm)
        a = np.exp(th[3:6].numpy().astype(np.float64))
        V = 4 / 3 * np.pi * np.prod(a)
        cc = th[6:].numpy().astype(np.float64) * 1000.0
        var_logV = float(Cov[3:6, 3:6].sum())
        var_c = np.diag(Cov)[6:] * 1e6
        # material coordinates (contrast relative to urine)
        if C == 2:
            Binv = np.linalg.inv(B)
            g = Binv @ cc
            gM = Binv @ M                                       # blur-invariant (scale-free) direction
            CovM = np.diag((sig_n[:C] ** 2) * Om.sum() * ETA)
            Jang = np.array([-gM[1], gM[0]]) / (gM @ gM + 1e-9)
            var_ang = float(Jang @ Binv @ CovM @ Binv.T @ Jang)
            ang = float(np.arctan2(gM[1], gM[0]))
            Cg = Binv @ np.diag(var_c) @ Binv.T
            mag = float(np.linalg.norm(g))
            var_logmag = float((g @ Cg @ g) / (mag ** 4 + 1e-9))
        else:
            ang, var_ang = 0.0, np.pi ** 2
            bref = B[0, 1] * 0.5 + B[0, 0] * 0.5
            mag = float(cc[0] / bref)
            var_logmag = float(var_c[0] / (cc[0] ** 2 + 1e-9))
        out.update(V=V, a=a, c=cc, var_logV=var_logV, var_c=var_c, chi=chi, z_twin=(chi - 1) * np.sqrt(nu / 2),
                   ang=ang, var_ang=var_ang, logmag=float(np.log(max(mag, 1e-3))), var_logmag=var_logmag,
                   dmax=float(2 * a.max()), centroid=(th[:3].numpy() + lo).tolist(), R=R)
        if feat is not None:
            w = occ * K
            out["emb"] = (feat[:, w > 0] * w[w > 0]).sum(1) / (w.sum() + 1e-6)
        inner = ndi.binary_erosion(K)
        out["het"] = float(img[-1][inner].std()) if inner.sum() > 3 else 0.0
        out["mask_k"] = K
        res.append(out)
    return res
