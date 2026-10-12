"""Evaluation of SDT-Net, its ablations and the comparison pipelines on every cohort.
Produces results/*.json consumed by the figure and manuscript builders."""
import os, json, pickle, itertools, time
import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import heads as H
import physics as P
from train import RUNS

OUT = os.environ.get("SDT_RES", "/home/claude/results")
EVAL = ["test", "extD", "extE"]
CL = H.CL
CIDX = H.CIDX


def load(v, s, tag=""):
    return pickle.load(open(f"{RUNS}/rec_{v}_s{s}{tag}.pkl", "rb"))


# ------------------------------------------------------------------ detection / segmentation
def detection(rec, split):
    S = [s for s in rec["scans"] if s["split"] == split and s["scan"] == 0]
    ngt = sum(s["n_gt"] for s in S)
    ntp = sum(s["n_tp"] for s in S)
    nfp = sum(s["n_fp"] for s in S)
    nfd = sum(s["n_fp_distractor"] for s in S)
    small = big = small_tp = big_tp = 0
    for s in S:
        for i, d in enumerate(s["gt_dmax"]):
            hit = (i + 1) in s["detected_ids"]
            if d < 4.0:
                small += 1; small_tp += hit
            else:
                big += 1; big_tp += hit
    dsc = [s["dice_stone"] for s in S if s.get("dice_stone") is not None and np.isfinite(s["dice_stone"])]
    out = dict(sens=ntp / max(ngt, 1), fp_per_scan=nfp / len(S), fp_distr_per_scan=nfd / len(S),
               sens_small=small_tp / max(small, 1), sens_big=big_tp / max(big, 1), dice=float(np.mean(dsc)),
               n_scans=len(S), n_gt=ngt)
    if "dice_cs" in S[0]:
        out["dice_cs"] = float(np.mean([s["dice_cs"] for s in S]))
    bg = [s["bg_mae"] for s in S if "bg_mae" in s]
    if bg:
        out["bg_mae"] = float(np.mean(bg))
    return out


# ------------------------------------------------------------------ classic heads
CLASSIC_LDA_IDX = [0, 1, 6, 7, 18, 19]          # mean/max per channel, DER, DE flag


class ClassicHead:
    def __init__(self, kind, seed=0):
        self.kind, self.seed = kind, seed

    def fit(self, rows):
        X = np.stack([q["feat"] for q in rows]); y = np.array([CIDX[q["cls"]] for q in rows])
        if self.kind == "lda":
            self.m = LinearDiscriminantAnalysis().fit(X[:, CLASSIC_LDA_IDX], y)
        else:
            self.m = RandomForestClassifier(400, min_samples_leaf=2, class_weight="balanced_subsample",
                                            random_state=self.seed, n_jobs=1).fit(X, y)
        self.T = 1.0
        return self

    def _p(self, rows):
        X = np.stack([q["feat"] for q in rows])
        p = self.m.predict_proba(X[:, CLASSIC_LDA_IDX] if self.kind == "lda" else X)
        full = np.full((len(rows), 6), 1e-6)
        full[:, self.m.classes_] = p
        lp = np.log(full) / self.T
        e = np.exp(lp - lp.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, rows):
        p = self._p(rows)
        return p, p[None]

    def calibrate_T(self, rows):
        y = np.array([CIDX[q["cls"]] for q in rows])
        best = (1e9, 1.0)
        for T_ in np.exp(np.linspace(-1.0, 1.5, 26)):
            self.T = T_
            p = self._p(rows)
            best = min(best, (-np.log(p[np.arange(len(y)), y] + 1e-9).mean(), T_))
        self.T = best[1]


# ------------------------------------------------------------------ helpers
def rows(rec, split, ood=False, scan0=None):
    """true-positive instances; evaluation cohorts use the first examination of each patient only
    (repeat examinations are reserved for the repeatability analysis)."""
    if scan0 is None:
        scan0 = split not in ("train", "val", "cal")
    return [q for q in rec["inst"] if q["split"] == split and q.get("tp") and bool(q.get("ood", False)) == ood
            and (not scan0 or q["scan"] == 0)]


def groups(R):
    return np.array([q["n_ch"] for q in R])


def ylab(R):
    return np.array([CIDX[q["cls"]] for q in R])


def entropy(p):
    return -(p * np.log(p + 1e-12)).sum(-1)


def mutual_info(P_):
    pm = P_.mean(0)
    return entropy(pm) - entropy(P_).mean(0)


def auroc(neg, pos):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return float(roc_auc_score(y, np.r_[neg, pos]))


def fpr_at_95(neg, pos):
    thr = np.quantile(pos, 0.05)
    return float((np.asarray(neg) >= thr).mean())


def burden_metrics(R, key):
    gt = np.array([q["gt_vol"] for q in R])
    est = np.array([key(q) for q in R])
    le = np.log(est / gt)
    small = np.array([q["gt_dmax"] < 4.0 for q in R])
    return dict(mdape=float(np.median(np.abs(est - gt) / gt) * 100), mae_log=float(np.mean(np.abs(le))),
                bias_log=float(np.mean(le)), loa_log=float(1.96 * np.std(le)),
                mdape_small=float(np.median(np.abs(est - gt)[small] / gt[small]) * 100) if small.any() else np.nan,
                mdape_big=float(np.median(np.abs(est - gt)[~small] / gt[~small]) * 100), n=len(R))


VOLKEYS = {"twin": lambda q: H.burden_estimates(q)["V_twin"], "seg": lambda q: H.burden_estimates(q)["V_seg"],
           "mom": lambda q: H.burden_estimates(q)["V_mom"], "fused": lambda q: H.burden_estimates(q)["V_fused"],
           "thr": lambda q: max(q["V_thr"], 0.05)}


def fit_vol_model(rec):
    """Gaussian variance model for the log-volume errors of twin and occupancy volumes, fitted by maximum
    likelihood on validation stones (the calibration split is kept for the conformal scale):
    var_t = k1^2 var_Laplace max(1, chi2) + k0^2, var_s = ss^2."""
    from scipy.optimize import minimize
    R = rows(rec, "val")
    et = np.array([np.log(max(q["V"], 0.05) / q["gt_vol"]) for q in R])
    es = np.array([np.log(max(q["V_seg"], 0.05) / q["gt_vol"]) for q in R])
    vl = np.array([max(q["var_logV"], 1e-6) * max(1.0, q["chi"]) for q in R])

    def nll(t):
        v = np.exp(2 * t[0]) * vl + np.exp(2 * t[1])
        return 0.5 * np.sum(np.log(v) + et ** 2 / v)
    t = minimize(nll, [0.0, np.log(0.05)], method="Nelder-Mead").x
    cal = dict(k1=float(np.exp(t[0])), k0=float(np.exp(t[1])), ss=float(np.sqrt(np.mean(es ** 2))))
    H.VOLCAL.update(cal)
    return cal


def fused_sd(q):
    return 1.0 / np.sqrt(1 / H.twin_var(q) + 1 / H.VOLCAL["ss"] ** 2)


def vol_conformal_q(rec, alpha=0.1):
    """split-conformal scale for the fused log-volume interval: quantile of |log error| / fused sd on calibration stones."""
    fit_vol_model(rec)
    R = rows(rec, "cal")
    r = np.array([abs(np.log(H.burden_estimates(q)["V_fused"] / q["gt_vol"])) / fused_sd(q) for q in R])
    n = len(r)
    return float(np.quantile(r, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0), method="higher"))


def interval_cover(R, qv=1.645):
    """coverage and median width (ratio upper/lower) of the fused log-volume interval  log V_f +- qv * sd."""
    c, w = [], []
    for q in R:
        sd = fused_sd(q)
        lv = np.log(H.burden_estimates(q)["V_fused"])
        c.append(abs(lv - np.log(q["gt_vol"])) <= qv * sd)
        w.append(np.exp(2 * qv * sd))
    return float(np.mean(c)), float(np.median(w))


# ------------------------------------------------------------------ one method evaluation
def eval_method(rec, head, conf_mondrian=True, ood_rec=True, alpha=0.1):
    tr, va, ca = rows(rec, "train"), rows(rec, "val"), rows(rec, "cal")
    head.fit(tr)
    head.calibrate_T(va)
    pc, _ = head.predict(ca)
    yc = ylab(ca)
    mond = H.Mondrian(alpha, conf_mondrian).fit(pc, yc, groups(ca))
    res = {}
    pred_store = {}
    for sp in EVAL:
        R = rows(rec, sp)
        p, Pm = head.predict(R)
        y = ylab(R)
        m = H.comp_metrics(p, y)
        S = mond.sets(p, groups(R))
        m["cov"] = float(S[np.arange(len(y)), y].mean())
        m["set"] = float(S.sum(1).mean())
        m["singleton"] = float((S.sum(1) == 1).mean())
        for g in (1, 2):
            gm = groups(R) == g
            if gm.any():
                m[f"cov_g{g}"] = float(S[gm][np.arange(gm.sum()), y[gm]].mean())
                m[f"set_g{g}"] = float(S[gm].sum(1).mean())
                m[f"f1_g{g}"] = H.comp_metrics(p[gm], y[gm])["macro_f1"]
        mixed = np.array([q.get("mixed", False) for q in R])
        if mixed.any():
            m["acc_mixed"] = float((p.argmax(1) == y)[mixed].mean())
            m["acc_pure"] = float((p.argmax(1) == y)[~mixed].mean())
            m["cov_mixed"] = float(S[mixed][np.arange(mixed.sum()), y[mixed]].mean())
        small = np.array([q["gt_dmax"] < 4.0 for q in R])
        m["acc_small"] = float((p.argmax(1) == y)[small].mean())
        m["acc_big"] = float((p.argmax(1) == y)[~small].mean())
        res[sp] = m
        pred_store[sp] = dict(p=p, y=y, S=S, Pm=Pm, keys=[(q["pid"], q["scan"], q["gt_id"]) for q in R],
                              n_ch=groups(R))
    return res, pred_store, mond


def ood_eval(rec, head, manifold=None):
    """BRU/XAN instances versus in-distribution test instances."""
    Rin = rows(rec, "test")
    Rood = rows(rec, "ood", ood=True)
    pin, Pin = head.predict(Rin)
    pout, Pout = head.predict(Rood)
    out = {}
    out["msp"] = auroc(-pin.max(1), -pout.max(1))
    out["entropy"] = auroc(entropy(pin), entropy(pout))
    if Pin.shape[0] > 1:
        out["mi"] = auroc(mutual_info(Pin), mutual_info(Pout))
    if manifold is not None:
        si = np.array([manifold.score(q) for q in Rin]); so = np.array([manifold.score(q) for q in Rood])
        out["manifold"] = auroc(si, so)
        out["manifold_fpr95"] = fpr_at_95(si, so)
        for g in (1, 2):
            gi = groups(Rin) == g; go = groups(Rood) == g
            if go.any():
                out[f"manifold_g{g}"] = auroc(si[gi], so[go])
                out[f"msp_g{g}"] = auroc(-pin.max(1)[gi], -pout.max(1)[go])
        out["_scores"] = (si.tolist(), so.tolist())
    out["n_in"], out["n_ood"] = len(Rin), len(Rood)
    if "emb" in Rin[0]:
        tr = rows(rec, "train")
        di = H.deep_maha(tr, Rin); do = H.deep_maha(tr, Rood)
        out["deep_maha"] = auroc(di, do)
    return out


def bootstrap_ci(vals_by_patient, stat=np.mean, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(vals_by_patient)
    arr = [np.asarray(vals_by_patient[k]) for k in keys]
    bs = []
    for _ in range(B):
        idx = rng.integers(len(keys), size=len(keys))
        bs.append(stat(np.concatenate([arr[i] for i in idx])))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def mcnemar(a, b):
    """exact McNemar on paired correctness vectors."""
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    return float(stats.binomtest(n01, n, 0.5).pvalue), n01, n10


# ------------------------------------------------------------------ treatment-level (dissolvable) referral
def ua_probs(p):
    """aggregate six-class probabilities into the treatment-relevant decision: uric acid (dissolvable) vs not."""
    return np.stack([p[:, CIDX["UA"]], 1 - p[:, CIDX["UA"]]], 1)


def binary_referral(rec, head, manifold=None, alpha=0.1, ood_q=0.95):
    tr, va, ca = rows(rec, "train"), rows(rec, "val"), rows(rec, "cal")
    head.fit(tr); head.calibrate_T(va)
    pc, _ = head.predict(ca)
    yc = (ylab(ca) != CIDX["UA"]).astype(int)
    M = H.Mondrian(alpha, True, score="lac").fit(ua_probs(pc), yc, groups(ca))
    tau = None
    if manifold is not None:
        tau = float(np.quantile([manifold.score(q) for q in ca], ood_q))
    out = {}
    for sp in EVAL + ["ood"]:
        R = rows(rec, sp, ood=(sp == "ood"))
        if not R:
            continue
        p, _ = head.predict(R)
        p2 = ua_probs(p)
        g = groups(R)
        S = M.sets(p2, g)
        single = S.sum(1) == 1
        flag = np.zeros(len(R), bool)
        if manifold is not None:
            flag = np.array([manifold.score(q) > tau for q in R])
        refer = ~single | flag
        pred = p2.argmax(1)
        if sp == "ood":
            # xanthine mimics uric acid; brushite mimics calcium stones. A silent failure is a confident,
            # unreferred call. For xanthine the dangerous call is "uric acid" (dissolution would be attempted).
            cls = np.array([q["cls"] for q in R])
            for c in ("XAN", "BRU"):
                m = cls == c
                out[f"ood:{c}"] = dict(n=int(m.sum()), refer=float(refer[m].mean()),
                                       silent_ua=float(((pred == 0) & ~refer)[m].mean()))
            continue
        y2 = (ylab(R) != CIDX["UA"]).astype(int)
        r = dict(cov=float(S[np.arange(len(R)), y2].mean()), refer=float(refer.mean()),
                 acc_kept=float((pred == y2)[~refer].mean()) if (~refer).any() else np.nan,
                 acc_all=float((pred == y2).mean()))
        for gg in (1, 2):
            m = g == gg
            if m.any():
                r[f"refer_g{gg}"] = float(refer[m].mean())
                r[f"cov_g{gg}"] = float(S[m][np.arange(m.sum()), y2[m]].mean())
                r[f"acc_kept_g{gg}"] = float((pred == y2)[m & ~refer].mean()) if (m & ~refer).any() else np.nan
        out[sp] = r
    return out
