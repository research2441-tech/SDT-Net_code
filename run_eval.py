"""Runs every evaluation of Section V and writes /home/claude/results/*.json."""
import os, json, pickle, sys, time
import numpy as np
from scipy import stats
import evaluate as E
import heads as H

R = E.OUT
os.makedirs(R, exist_ok=True)
SEEDS = [1, 2, 3]
ABL_SEEDS = [1, 2]
EVAL = E.EVAL


def have(v, s, tag=""):
    return os.path.exists(f"{E.RUNS}/rec_{v}_s{s}{tag}.pkl")


def jdump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(obj, open(f"{R}/{name}.json", "w"), indent=1, default=conv)


def agg(list_of_dicts):
    """mean and sd over seeds of nested metric dicts {split: {metric: v}}."""
    out = {}
    for sp in list_of_dicts[0]:
        out[sp] = {}
        for k in list_of_dicts[0][sp]:
            vals = [d[sp][k] for d in list_of_dicts if k in d[sp] and d[sp][k] is not None]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                out[sp][k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


COMP_METHODS = {          # name: (record variant, head factory)
    "HU-LDA": ("classic", lambda s: E.ClassicHead("lda", s)),
    "HU-RF": ("classic", lambda s: E.ClassicHead("rf", s)),
    "MT-UNet": ("mtunet", lambda s: H.Head("deep", seed=s)),
    "AG-UNet": ("agunet", lambda s: H.Head("hybrid", seed=s)),
    "SDT-Net": ("full", lambda s: H.Head("sdt", seed=s)),
}


def rec_of(v, s, tag=""):
    if v == "classic":
        return pickle.load(open(f"{E.RUNS}/rec_classic.pkl", "rb"))
    rec = E.load(v, s, tag)
    E.fit_vol_model(rec)            # calibrated variance model of this backbone (calibration split only)
    return rec


def main_comparison():
    det, comp, store, bur = {}, {}, {}, {}
    for name, (v, mk) in COMP_METHODS.items():
        seeds = [s for s in SEEDS if v == "classic" or have(v, s)]
        if not seeds:
            continue
        cres, dres = [], []
        for s in seeds:
            rec = rec_of(v, s)
            head = mk(s)
            res, ps, _ = E.eval_method(rec, head)
            cres.append(res)
            store[(name, s)] = ps
            dres.append({sp: E.detection(rec, sp) for sp in EVAL})
            if name in ("HU-RF", "MT-UNet", "AG-UNet", "SDT-Net"):
                keys = {"HU-RF": ["thr"], "MT-UNet": ["seg"], "AG-UNet": ["seg"], "SDT-Net": ["seg", "mom", "twin", "fused"]}[name]
                for kk in keys:
                    bur.setdefault(f"{name}:{kk}", []).append({sp: E.burden_metrics(E.rows(rec, sp), E.VOLKEYS[kk]) for sp in EVAL})
                if name == "SDT-Net":
                    qv = E.vol_conformal_q(rec)
                    iv = {}
                    for sp in EVAL:
                        cv, wd = E.interval_cover(E.rows(rec, sp), qv)
                        cn, wn = E.interval_cover(E.rows(rec, sp), 1.645)
                        iv[sp] = {"cover": cv, "width": wd, "cover_gauss": cn, "width_gauss": wn, "qv": qv}
                    bur.setdefault("SDT-Net:interval90", []).append(iv)
            print(name, s, {sp: round(res[sp]["macro_f1"], 3) for sp in EVAL}, flush=True)
        comp[name] = agg(cres)
        det[name] = agg(dres)
    jdump(comp, "comp_main"); jdump(det, "detection"); jdump({k: agg(v) for k, v in bur.items()}, "burden")
    pickle.dump(store, open(f"{R}/pred_store.pkl", "wb"))
    return store


def ablations():
    out = {}
    # head ablations on the SDT-Net backbone
    for fs in ["sdt", "notex", "nomat", "noprop", "mom"]:
        rs = []
        for s in ABL_SEEDS:
            if have("full", s):
                rs.append(E.eval_method(rec_of("full", s), H.Head(fs, seed=s))[0])
        if rs:
            out[f"head:{fs}"] = agg(rs)
    rs = []
    for s in ABL_SEEDS:
        if have("full", s):
            rs.append(E.eval_method(E.load("full", s), H.Head("sdt", seed=s), conf_mondrian=False)[0])
    if rs:
        out["conf:pooled"] = agg(rs)
    for v in ["nossl", "noanat", "nobg"]:
        rs, ds, bs = [], [], []
        for s in ABL_SEEDS:
            if have(v, s):
                rec = rec_of(v, s)
                rs.append(E.eval_method(rec, H.Head("sdt", seed=s))[0])
                ds.append({sp: E.detection(rec, sp) for sp in EVAL})
                bs.append({sp: E.burden_metrics(E.rows(rec, sp), E.VOLKEYS["fused"]) for sp in EVAL})
        if rs:
            out[f"bb:{v}"] = agg(rs); out[f"bbdet:{v}"] = agg(ds); out[f"bbbur:{v}"] = agg(bs)
    jdump(out, "ablation")


def ood():
    out = {}
    for s in SEEDS:
        if not have("full", s):
            continue
        rec = rec_of("full", s)
        head = H.Head("sdt", seed=s).fit(E.rows(rec, "train"))
        head.calibrate_T(E.rows(rec, "val"))
        man = H.MaterialManifold().fit(E.rows(rec, "train"))
        out.setdefault("SDT-Net", []).append(E.ood_eval(rec, head, man))
        for name, v, fs in [("MT-UNet", "mtunet", "deep"), ("AG-UNet", "agunet", "hybrid")]:
            if have(v, s):
                r2 = E.load(v, s)
                h2 = H.Head(fs, seed=s).fit(E.rows(r2, "train")); h2.calibrate_T(E.rows(r2, "val"))
                out.setdefault(name, []).append(E.ood_eval(r2, h2, None))
    summ = {}
    for k, L in out.items():
        summ[k] = {m: (float(np.mean([d[m] for d in L])), float(np.std([d[m] for d in L])))
                   for m in L[0] if not m.startswith("_") and isinstance(L[0][m], float)}
    summ["_scores_s1"] = out["SDT-Net"][0]["_scores"]
    jdump(summ, "ood")


def referral(store):
    """risk-coverage: SDT-Net ordered by (set size, OOD distance, confidence); baselines by max probability."""
    out = {}
    for (name, s), ps in store.items():
        for sp in EVAL:
            p, y, S = ps[sp]["p"], ps[sp]["y"], ps[sp]["S"]
            corr = p.argmax(1) == y
            score = -p.max(1)
            if name == "SDT-Net":
                score = S.sum(1) * 10 - p.max(1)
            o = np.argsort(score, kind="stable")
            n = len(o)
            cov = np.arange(1, n + 1) / n
            risk = np.cumsum(~corr[o]) / np.arange(1, n + 1)
            aurc = float(np.mean(risk))
            keep = S.sum(1) == 1
            out.setdefault(name, {}).setdefault(sp, []).append(dict(aurc=aurc, risk=risk[::max(1, n // 50)].tolist(),
                                                                     cov=cov[::max(1, n // 50)].tolist(),
                                                                     single_rate=float(keep.mean()),
                                                                     single_acc=float(corr[keep].mean()) if keep.any() else np.nan))
    summ = {}
    for name, d in out.items():
        summ[name] = {}
        for sp, L in d.items():
            summ[name][sp] = dict(aurc=(float(np.mean([x["aurc"] for x in L])), float(np.std([x["aurc"] for x in L]))),
                                  single_rate=float(np.mean([x["single_rate"] for x in L])),
                                  single_acc=float(np.nanmean([x["single_acc"] for x in L])),
                                  curve=(L[0]["cov"], L[0]["risk"]))
    jdump(summ, "referral")


def paired_tests(store):
    """McNemar on per-stone correctness and Wilcoxon on |log volume error| against each comparator (seed 1),
    restricted to stones detected by both pipelines."""
    out = {}
    recs = {n: rec_of(COMP_METHODS[n][0], 1) for n in COMP_METHODS}
    for sp in EVAL:
        ref = store[("SDT-Net", 1)][sp]
        rk = {k: i for i, k in enumerate(ref["keys"])}
        res = {}
        pv = []
        for name in ["HU-LDA", "HU-RF", "MT-UNet", "AG-UNet"]:
            if (name, 1) not in store:
                continue
            o = store[(name, 1)][sp]
            ok = {k: i for i, k in enumerate(o["keys"])}
            common = [k for k in rk if k in ok]
            a = np.array([ref["p"][rk[k]].argmax() == ref["y"][rk[k]] for k in common])
            b = np.array([o["p"][ok[k]].argmax() == o["y"][ok[k]] for k in common])
            p, n01, n10 = E.mcnemar(a, b)
            res[name] = dict(n=len(common), acc_sdt=float(a.mean()), acc_other=float(b.mean()), p=p, n01=n01, n10=n10)
            pv.append(p)
        adj = E.holm(pv)
        for (name, r), pa in zip(res.items(), adj):
            r["p_holm"] = float(pa)
        out[f"comp:{sp}"] = res
        # burden
        rows_s = {(q["pid"], q["scan"], q["gt_id"]): q for q in E.rows(recs["SDT-Net"], sp)}
        res, pv = {}, []
        for name, kk in [("HU-RF", "thr"), ("MT-UNet", "seg"), ("AG-UNet", "seg"), ("SDT-Net seg", "seg")]:
            rr = recs["SDT-Net"] if name == "SDT-Net seg" else recs[name]
            rows_o = {(q["pid"], q["scan"], q["gt_id"]): q for q in E.rows(rr, sp)}
            common = [k for k in rows_s if k in rows_o]
            ea = np.array([abs(np.log(E.VOLKEYS["fused"](rows_s[k]) / rows_s[k]["gt_vol"])) for k in common])
            eb = np.array([abs(np.log(E.VOLKEYS[kk](rows_o[k]) / rows_o[k]["gt_vol"])) for k in common])
            w = stats.wilcoxon(ea, eb)
            res[name] = dict(n=len(common), med_sdt=float(np.median(ea)), med_other=float(np.median(eb)), p=float(w.pvalue))
            pv.append(float(w.pvalue))
        for (name, r), pa in zip(res.items(), E.holm(pv)):
            r["p_holm"] = float(pa)
        out[f"vol:{sp}"] = res
    # bootstrap CI of SDT-Net macro-F1 and of the F1 difference with AG-UNet (seed 1)
    from sklearn.metrics import f1_score
    for sp in EVAL:
        ref = store[("SDT-Net", 1)][sp]
        pids = np.array([k[0] for k in ref["keys"]])
        up = np.unique(pids)
        rng = np.random.default_rng(0)
        f1s = []
        o = store.get(("AG-UNet", 1), {}).get(sp)
        ok = {k: i for i, k in enumerate(o["keys"])} if o else {}
        common = [i for i, k in enumerate(ref["keys"]) if k in ok]
        diffs = []
        for _ in range(2000):
            bp = rng.choice(up, len(up))
            idx = np.concatenate([np.where(pids == p)[0] for p in bp])
            f1s.append(f1_score(ref["y"][idx], ref["p"][idx].argmax(1), average="macro", labels=range(6)))
            if o:
                ci = [i for i in idx if ref["keys"][i] in ok]
                oi = [ok[ref["keys"][i]] for i in ci]
                diffs.append(f1_score(ref["y"][ci], ref["p"][ci].argmax(1), average="macro", labels=range(6)) -
                             f1_score(o["y"][oi], o["p"][oi].argmax(1), average="macro", labels=range(6)))
        out[f"boot:{sp}"] = dict(f1_ci=[float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
                                 diff_ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else None)
    jdump(out, "paired")


def repeatability():
    out = {}
    for name, v, keys in [("SDT-Net", "full", ["fused", "seg", "twin"]), ("HU-RF", "classic", ["thr"]),
                          ("MT-UNet", "mtunet", ["seg"])]:
        rec = rec_of(v, 1)
        R0 = {(q["pid"], q["gt_id"]): q for q in E.rows(rec, "test", scan0=False) if q["scan"] == 0}
        R1 = {(q["pid"], q["gt_id"]): q for q in E.rows(rec, "test", scan0=False) if q["scan"] == 1}
        common = [k for k in R0 if k in R1]
        for kk in keys:
            a = np.array([E.VOLKEYS[kk](R0[k]) for k in common]); b = np.array([E.VOLKEYS[kk](R1[k]) for k in common])
            wcv = float(np.sqrt(np.mean(((a - b) ** 2) / 2 / ((a + b) / 2) ** 2)) * 100)
            out[f"{name}:{kk}"] = dict(n=len(common), wcv=wcv, rc=float(1.96 * np.sqrt(2) * np.std(np.log(a / b)) / np.sqrt(2)))
    jdump(out, "repeat")


# ---------------------------------------------------------------- additional analyses
def eval_cross(rec_fit, rec_eval, head, alpha=0.1):
    """fit head/temperature/conformal on rec_fit, evaluate on the evaluation cohorts of rec_eval."""
    head.fit(E.rows(rec_fit, "train")); head.calibrate_T(E.rows(rec_fit, "val"))
    ca = E.rows(rec_fit, "cal"); pc, _ = head.predict(ca)
    mond = H.Mondrian(alpha, True).fit(pc, E.ylab(ca), E.groups(ca))
    out = {}
    for sp in EVAL:
        R = E.rows(rec_eval, sp)
        if not R:
            continue
        p, _ = head.predict(R); y = E.ylab(R)
        m = H.comp_metrics(p, y)
        S = mond.sets(p, E.groups(R))
        m["cov"] = float(S[np.arange(len(y)), y].mean()); m["set"] = float(S.sum(1).mean())
        m.update({f"vol_{k}": v for k, v in E.burden_metrics(R, E.VOLKEYS["fused"]).items()})
        out[sp] = m
    return out


def bg_and_twin_ablation():
    out = {}
    for tag, lab in [("_bgnet", "learned residual counterfactual"), ("_bgann", "annulus background"),
                     ("_noocc", "no occupancy floor")]:
        rs, bs, ds = [], [], []
        for s in ABL_SEEDS:
            if have("full", s, tag):
                rec = rec_of("full", s, tag)
                rs.append(E.eval_method(rec, H.Head("sdt", seed=s))[0])
                bs.append({sp: E.burden_metrics(E.rows(rec, sp), E.VOLKEYS["fused"]) for sp in EVAL})
                ds.append({sp: E.burden_metrics(E.rows(rec, sp), E.VOLKEYS["twin"]) for sp in EVAL})
                bm = [x.get("bg_mae", np.nan) for x in rec["scans"] if x["split"] == "test" and x["scan"] == 0]
                out.setdefault(f"{tag}:bg_mae", []).append(float(np.nanmean(bm)))
        if rs:
            out[tag] = dict(label=lab, comp=agg(rs), fused=agg(bs), twin=agg(ds), seeds=len(rs))
    # background error of the reference configuration
    for s in ABL_SEEDS:
        if have("full", s):
            rec = E.load("full", s)
            out.setdefault("ref:bg_mae", []).append(float(np.nanmean([x.get("bg_mae", np.nan) for x in rec["scans"]
                                                                      if x["split"] == "test" and x["scan"] == 0])))
            out.setdefault("ref:bg_mae_net", []).append(float(np.nanmean([x.get("bg_mae_net", np.nan) for x in rec["scans"]
                                                                          if x["split"] == "test" and x["scan"] == 0])))
    jdump(out, "twin_ablation")


def psf_sensitivity():
    out = {}
    if not have("full", 1):
        return
    base = rec_of("full", 1)
    out["1.00"] = eval_cross(base, base, H.Head("sdt", seed=1))
    for tag, sc in [("_psf080", "0.80"), ("_psf125", "1.25")]:
        if have("full", 1, tag):
            ev = E.load("full", 1, tag)
            out[sc] = eval_cross(base, ev, H.Head("sdt", seed=1))
    jdump(out, "psf")


def alpha_sweep_and_classes(store):
    out = {}
    from sklearn.metrics import confusion_matrix, f1_score
    for s in SEEDS:
        if not have("full", s):
            continue
        rec = rec_of("full", s)
        head = H.Head("sdt", seed=s).fit(E.rows(rec, "train")); head.calibrate_T(E.rows(rec, "val"))
        ca = E.rows(rec, "cal"); pc, _ = head.predict(ca)
        for a in (0.05, 0.1, 0.2):
            for mond in (True, False):
                M = H.Mondrian(a, mond).fit(pc, E.ylab(ca), E.groups(ca))
                for sp in EVAL:
                    R = E.rows(rec, sp); p, _ = head.predict(R); y = E.ylab(R); g = E.groups(R); S = M.sets(p, g)
                    for gg in (1, 2):
                        m = g == gg
                        if m.any():
                            key = f"{a}:{'mond' if mond else 'pool'}:{sp}:g{gg}"
                            out.setdefault(key, []).append((float(S[m][np.arange(m.sum()), y[m]].mean()), float(S[m].sum(1).mean())))
    summ = {k: dict(cov=float(np.mean([x[0] for x in v])), set=float(np.mean([x[1] for x in v]))) for k, v in out.items()}
    # per-class F1 and confusion (seed 1) for SDT-Net and the deep baselines on internal test, dual energy
    for name in ["SDT-Net", "AG-UNet", "MT-UNet", "HU-RF"]:
        if (name, 1) not in store:
            continue
        for sp in EVAL:
            ps = store[(name, 1)][sp]
            for gg in (1, 2):
                m = ps["n_ch"] == gg
                if m.any():
                    summ[f"cm:{name}:{sp}:g{gg}"] = confusion_matrix(ps["y"][m], ps["p"][m].argmax(1), labels=range(6)).tolist()
                    summ[f"f1c:{name}:{sp}:g{gg}"] = f1_score(ps["y"][m], ps["p"][m].argmax(1), labels=range(6), average=None,
                                                              zero_division=0).tolist()
                    summ[f"uaauc:{name}:{sp}:g{gg}"] = float(E.roc_auc_score(ps["y"][m] == 0, ps["p"][m][:, 0])) \
                        if 0 < (ps["y"][m] == 0).sum() < m.sum() else None
    jdump(summ, "conformal_classes")


def size_bins():
    bins = [(0, 3), (3, 4), (4, 6), (6, 10), (10, 99)]
    out = {}
    for name, v, kk in [("HU threshold", "classic", "thr"), ("MT-UNet seg", "mtunet", "seg"), ("AG-UNet seg", "agunet", "seg"),
                        ("SDT-Net seg", "full", "seg"), ("SDT-Net twin", "full", "twin"), ("SDT-Net fused", "full", "fused")]:
        res = []
        for s in SEEDS:
            if v != "classic" and not have(v, s):
                continue
            rec = rec_of(v, s)
            r = {}
            for sp in EVAL:
                R = E.rows(rec, sp)
                for lo, hi in bins:
                    Rb = [q for q in R if lo <= q["gt_dmax"] < hi]
                    if Rb:
                        e = [abs(E.VOLKEYS[kk](q) - q["gt_vol"]) / q["gt_vol"] * 100 for q in Rb]
                        r[f"{sp}:{lo}-{hi}"] = float(np.median(e))
            res.append(r)
            if v == "classic":
                break
        out[name] = {k: (float(np.mean([d[k] for d in res if k in d])), float(np.std([d[k] for d in res if k in d])))
                     for k in res[0]}
    # scatter data for the Bland-Altman figure (seed 1, internal test)
    sc = {}
    for name, v, kk in [("HU threshold", "classic", "thr"), ("MT-UNet seg", "mtunet", "seg"), ("SDT-Net fused", "full", "fused")]:
        rec = rec_of(v, 1)
        R = E.rows(rec, "test")
        sc[name] = [(q["gt_vol"], E.VOLKEYS[kk](q)) for q in R]
    out["_scatter"] = sc
    jdump(out, "size_bins")


def material_plane():
    rec = E.load("full", 1)
    pts = {}
    for sp in ["test", "ood"]:
        for q in E.rows(rec, sp, ood=(sp == "ood")):
            if q["n_ch"] == 2:
                pts.setdefault(q["cls"], []).append((q["ang"], q["logmag"], np.sqrt(q["var_ang"]), q["gt_dmax"], q.get("mixed", False)))
    jdump(pts, "material_plane")


def efficiency():
    import torch
    from model import SDTBackbone, n_params
    out = {}
    for v in ["full", "mtunet", "agunet"]:
        js = [json.load(open(f"{E.RUNS}/{v}_s{s}.pt.json")) for s in SEEDS if os.path.exists(f"{E.RUNS}/{v}_s{s}.pt.json")]
        if js:
            out[v] = dict(params=js[0]["params"], train_min=float(np.mean([j["train_s"] for j in js]) / 60))
    jdump(out, "efficiency")




def treatment_referral():
    out = {}
    for name, (v, mk) in COMP_METHODS.items():
        for s in SEEDS:
            if v != "classic" and not have(v, s):
                continue
            rec = rec_of(v, s)
            man = H.MaterialManifold().fit(E.rows(rec, "train")) if name == "SDT-Net" else None
            out.setdefault(name, []).append(E.binary_referral(rec, mk(s), man))
            if name == "SDT-Net":
                out.setdefault("SDT-Net (no OOD flag)", []).append(E.binary_referral(rec, mk(s), None))
    summ = {}
    for name, L in out.items():
        summ[name] = {}
        for sp in L[0]:
            summ[name][sp] = {k: (float(np.nanmean([d[sp][k] for d in L])), float(np.nanstd([d[sp][k] for d in L])))
                              for k in L[0][sp]}
    jdump(summ, "treatment_referral")


if __name__ == "__main__":
    what = sys.argv[1:] or ["main", "abl", "ood", "ref", "paired", "rep", "twinabl", "psf", "alpha", "size", "plane", "eff", "treat"]
    store = None
    if "main" in what:
        store = main_comparison()
    if store is None and os.path.exists(f"{R}/pred_store.pkl"):
        store = pickle.load(open(f"{R}/pred_store.pkl", "rb"))
    if "abl" in what:
        ablations()
    if "ood" in what:
        ood()
    if "ref" in what:
        referral(store)
    if "paired" in what:
        paired_tests(store)
    if "rep" in what:
        repeatability()
    if "twinabl" in what:
        bg_and_twin_ablation()
    if "psf" in what:
        psf_sensitivity()
    if "alpha" in what:
        alpha_sweep_and_classes(store)
    if "size" in what:
        size_bins()
    if "plane" in what:
        material_plane()
    if "eff" in what:
        efficiency()
    if "treat" in what:
        treatment_referral()


