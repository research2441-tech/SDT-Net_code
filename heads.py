"""Composition heads, measurement-propagated prediction, Mondrian conformal sets,
material-manifold out-of-distribution score, selective referral and burden estimation."""
import os, pickle, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import physics as P
import twin as T
from train import RUNS

CL = P.CLASSES
CIDX = {c: i for i, c in enumerate(CL)}
FEATSETS = ["sdt", "notex", "nomat", "noprop", "mom", "deep", "hybrid"]


def basis_coords(q, c):
    B, _ = T.basis(q["site"], round(q["wed"]))
    if q["n_ch"] == 2:
        g = np.linalg.solve(B, c)
        return float(np.log(max(np.linalg.norm(g), 1e-3)))
    return float(np.log(max(c[0] / (0.5 * B[0, 0] + 0.5 * B[0, 1]), 1e-3)))


def features(q, fs):
    """returns (mu, sd) where sd marks the measurement-uncertain coordinates (zero elsewhere)."""
    dect = float(q["n_ch"] == 2)
    kv = P.PROTOCOLS[q["site"]]["ch"][0][0] / 150.0
    emb = list(np.asarray(q["emb"], float))
    if fs in ("deep", "hybrid"):
        hm = np.asarray(q["hu_mean"], float) / 1000.0
        hx = np.asarray(q["hu_max"], float) / 1000.0
        if len(hm) == 1:
            hm, hx = np.r_[hm, hm], np.r_[hx, hx]
        mu = emb + list(hm) + list(hx) + [np.log(max(q["V_seg"], 0.1)), dect, kv]
        if fs == "hybrid":
            mu += [q["der"]]
        return np.array(mu), np.zeros(len(mu))
    ang, sang = q["ang"], np.sqrt(q["var_ang"])
    if fs == "mom":
        lm = basis_coords(q, np.asarray(q["c_mom"], float))
        lv = np.log(max(q["V_mom"], 0.1))
    else:
        lm, lv = q["logmag"], np.log(max(q["V"], 0.1))
    slm, slv = np.sqrt(min(q["var_logmag"], 25.0)), np.sqrt(min(q["var_logV"], 25.0))
    if fs == "nomat":
        c = np.asarray(q["c"], float) / 1000.0
        M = np.asarray(q["M"], float)
        c2 = np.r_[c, c] if len(c) == 1 else c
        ratio = M[0] / M[1] if len(M) == 2 and abs(M[1]) > 1e-6 else 0.0
        base = [c2[0], c2[1], ratio, lv]
        sd = [0, 0, 0, slv]
    else:
        base = [np.sin(ang) * dect, np.cos(ang) * dect, lm, lv]
        sd = [sang * dect, sang * dect, slm, slv]
    unc = [min(sang, np.pi) * dect + np.pi * (1 - dect), slm, slv, np.tanh(q["chi"] / 50.0)]
    mu = base + unc + [dect, kv]
    sd = sd + [0] * (len(unc) + 2)
    if fs not in ("notex",):
        mu += emb + [q["het"] / 1000.0]
        sd += [0] * (len(emb) + 1)
    return np.array(mu), np.array(sd)


class MLP(nn.Module):
    def __init__(self, d, h=64, k=6, p=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(), nn.Linear(h, k))

    def forward(self, x):
        return self.f(x)


def tp_rows(rec, split, ood=False):
    return [q for q in rec["inst"] if q["split"] == split and q.get("tp") and (q.get("ood", False) == ood)]


class Head:
    def __init__(self, fs, n_ens=5, prop=True, seed=0):
        self.fs, self.n_ens, self.prop, self.seed = fs, n_ens, prop and fs not in ("deep", "hybrid", "noprop"), seed

    def _mat(self, rows):
        A = [features(q, self.fs) for q in rows]
        return np.stack([a[0] for a in A]), np.stack([a[1] for a in A])

    def fit(self, rows, epochs=400):
        mu, sd = self._mat(rows)
        y = np.array([CIDX[q["cls"]] for q in rows])
        self.m, self.s = mu.mean(0), mu.std(0) + 1e-6
        cnt = np.bincount(y, minlength=6) + 1
        w = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float32)
        w = w / w.mean()
        X = torch.tensor((mu - self.m) / self.s, dtype=torch.float32)
        Sd = torch.tensor(sd / self.s, dtype=torch.float32)
        Y = torch.tensor(y)
        self.nets = []
        for e in range(self.n_ens):
            torch.manual_seed(1000 * self.seed + e)
            net = MLP(X.shape[1])
            opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3)
            idx = torch.randint(0, len(X), (len(X),))          # bootstrap per member
            for ep in range(epochs):
                xb = X[idx]
                if self.prop:
                    xb = xb + torch.randn_like(xb) * Sd[idx]
                loss = F.cross_entropy(net(xb), Y[idx], weight=w, label_smoothing=0.02)
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            self.nets.append(net)
        self.T = 1.0
        return self

    def logits_members(self, rows, S=32):
        mu, sd = self._mat(rows)
        X = torch.tensor((mu - self.m) / self.s, dtype=torch.float32)
        Sd = torch.tensor(sd / self.s, dtype=torch.float32)
        out = []
        with torch.no_grad():
            for net in self.nets:
                if self.prop:
                    xs = X[None] + torch.randn(S, *X.shape, generator=torch.Generator().manual_seed(7)) * Sd[None]
                    p = torch.softmax(net(xs) / self.T, -1).mean(0)
                else:
                    p = torch.softmax(net(X) / self.T, -1)
                out.append(p.numpy())
        return np.stack(out)                                   # (E, N, K)

    def predict(self, rows):
        P_ = self.logits_members(rows)
        return P_.mean(0), P_

    def calibrate_T(self, rows):
        y = np.array([CIDX[q["cls"]] for q in rows])
        best = (1e9, 1.0)
        for T_ in np.exp(np.linspace(-1.0, 1.2, 23)):
            self.T = T_
            p, _ = self.predict(rows)
            nll = -np.log(p[np.arange(len(y)), y] + 1e-9).mean()
            best = min(best, (nll, T_))
        self.T = best[1]
        return self.T


# ------------------------- metrics -------------------------
def ece(p, y, nb=10):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    for b in range(nb):
        m = (conf > b / nb) & (conf <= (b + 1) / nb)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def comp_metrics(p, y):
    from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
    pred = p.argmax(1)
    yo = np.eye(6)[y]
    ua = y == CIDX["UA"]
    out = dict(n=len(y), acc=(pred == y).mean(), macro_f1=f1_score(y, pred, average="macro", labels=range(6)),
               bacc=balanced_accuracy_score(y, pred), ece=ece(p, y), brier=((p - yo) ** 2).sum(1).mean(),
               nll=-np.log(p[np.arange(len(y)), y] + 1e-9).mean())
    out["ua_auc"] = roc_auc_score(ua, p[:, CIDX["UA"]]) if 0 < ua.sum() < len(y) else np.nan
    return out


# ------------------------- conformal -------------------------
def aps_scores(p, y):
    o = np.argsort(-p, 1)
    cs = np.cumsum(np.take_along_axis(p, o, 1), 1)
    rank = np.argsort(o, 1)[np.arange(len(y)), y]
    return cs[np.arange(len(y)), rank]


def aps_sets(p, qhat):
    o = np.argsort(-p, 1)
    cs = np.cumsum(np.take_along_axis(p, o, 1), 1)
    k = (cs < qhat).sum(1) + 1
    S = np.zeros_like(p, bool)
    for i in range(len(p)):
        S[i, o[i, :k[i]]] = True
    return S


def conformal_q(scores, alpha):
    n = len(scores)
    lvl = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, lvl, method="higher"))


def lac_scores(p, y):
    return 1.0 - p[np.arange(len(y)), y]


def lac_sets(p, qhat):
    return p >= 1.0 - qhat


class Mondrian:
    """class of conformal calibration keyed by spectral availability (dual vs single energy).
    score='aps' (adaptive sets, six-class composition) or 'lac' (threshold sets, binary treatment decision)."""

    def __init__(self, alpha=0.1, mondrian=True, score="aps"):
        self.alpha, self.mondrian, self.score = alpha, mondrian, score

    def fit(self, p, y, groups):
        self.q = {}
        keys = np.unique(groups) if self.mondrian else ["all"]
        for g in keys:
            m = groups == g if self.mondrian else np.ones(len(y), bool)
            sc = aps_scores(p[m], y[m]) if self.score == "aps" else lac_scores(p[m], y[m])
            self.q[g] = conformal_q(sc, self.alpha)
        return self

    def sets(self, p, groups):
        S = np.zeros_like(p, bool)
        for g, q in self.q.items():
            m = groups == g if self.mondrian else np.ones(len(p), bool)
            if m.any():
                S[m] = aps_sets(p[m], q) if self.score == "aps" else lac_sets(p[m], q)
        return S


# ------------------------- material-manifold OOD score -------------------------
class MaterialManifold:
    def fit(self, rows):
        self.st = {}
        for d in (1, 2):
            for c in CL:
                R = [q for q in rows if q["cls"] == c and q["n_ch"] == d]
                if len(R) < 3:
                    continue
                Z = np.array([self._z(q) for q in R])
                self.st[(d, c)] = (Z.mean(0), np.cov(Z.T) + 1e-4 * np.eye(Z.shape[1]) if Z.shape[1] > 1
                                   else np.atleast_2d(Z.var() + 1e-4))
        return self

    @staticmethod
    def _z(q):
        if q["n_ch"] == 2:
            return np.array([q["ang"], q["logmag"]])
        return np.array([q["logmag"]])

    @staticmethod
    def _S(q):
        if q["n_ch"] == 2:
            return np.diag([q["var_ang"], min(q["var_logmag"], 25.0)])
        return np.atleast_2d(min(q["var_logmag"], 25.0))

    def score(self, q):
        d = q["n_ch"]
        z, Sm = self._z(q), self._S(q)
        best = 1e9
        for (dd, c), (m, S) in self.st.items():
            if dd != d:
                continue
            r = z - m
            best = min(best, float(r @ np.linalg.solve(S + Sm, r)))
        return best


def deep_maha(train_rows, rows):
    E = np.array([q["emb"] for q in train_rows])
    y = np.array([CIDX[q["cls"]] for q in train_rows])
    mus = {k: E[y == k].mean(0) for k in range(6) if (y == k).sum() > 2}
    Sc = sum(np.cov((E[y == k] - mus[k]).T) * (y == k).sum() for k in mus) / len(E) + 1e-3 * np.eye(E.shape[1])
    Si = np.linalg.inv(Sc)
    out = []
    for q in rows:
        e = np.asarray(q["emb"])
        out.append(min(float((e - m) @ Si @ (e - m)) for m in mus.values()))
    return np.array(out)


# ------------------------- burden -------------------------
VOLCAL = dict(k1=1.0, k0=0.0, ss=0.25)       # variance model, refitted on calibration stones (evaluate.fit_vol_model)


def twin_var(q, cal=None):
    c = cal or VOLCAL
    return c["k1"] ** 2 * max(q["var_logV"], 1e-6) * max(1.0, q["chi"]) + c["k0"] ** 2


def burden_estimates(q, cal=None):
    c = cal or VOLCAL
    Vt, vt = max(q["V"], 0.05), twin_var(q, c)
    Vs = max(q["V_seg"], 0.05)
    w1, w2 = 1 / vt, 1 / c["ss"] ** 2
    Vf = np.exp((w1 * np.log(Vt) + w2 * np.log(Vs)) / (w1 + w2))
    return dict(V_twin=Vt, V_seg=Vs, V_mom=max(q["V_mom"], 0.05), V_fused=Vf)
