"""Failure analysis by stone structure (SDT-Net, seeds 1-3, internal test)."""
import json, numpy as np, evaluate as E, heads as H
out = {}
for s in (1, 2, 3):
    rec = E.load("full", s); E.fit_vol_model(rec)
    R = E.rows(rec, "test")
    for mode in ("pure", "homog", "core"):
        Rs = [q for q in R if q["mode"] == mode]
        t = np.array([abs(q["V"] / q["gt_vol"] - 1) for q in Rs])
        f = np.array([abs(H.burden_estimates(q)["V_fused"] / q["gt_vol"] - 1) for q in Rs])
        b = np.array([np.log(q["V"] / q["gt_vol"]) for q in Rs])
        out.setdefault(mode, []).append(dict(n=len(Rs), twin=float(np.median(t)) * 100, fused=float(np.median(f)) * 100,
                                             bias=float(np.median(b)), chi=float(np.median([q["chi"] for q in Rs]))))
summ = {m: {k: float(np.mean([d[k] for d in L])) for k in L[0]} for m, L in out.items()}
json.dump(summ, open("/home/claude/results/failure.json", "w"), indent=1)
print(summ)
