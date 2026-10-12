"""Cohort statistics for Table II (computed from the stored ground truth)."""
import json, collections, numpy as np
import data as Dt
J = Dt.jobs()
out = {}
for sp in ["train", "val", "cal", "test", "extD", "extE", "ood", "unl"]:
    js = [j for j in J if j["split"] == sp]
    pts = len({j["pid"] for j in js})
    rows = dict(exams=len(js), patients=pts, rescans=sum(j["scan"] == 1 for j in js),
                sites="".join(sorted({j["site"] for j in js})))
    if sp != "unl":
        st, wed, hyd, ncal, n0 = [], [], 0, 0, 0
        for j in js:
            m = json.loads(str(np.load(Dt.case_path(j))["meta"]))
            if j["scan"] == 0:
                st += m["stones"]; wed.append(m["wed"]); hyd += m["hydro"]; ncal += m["n_calc"] > 0; n0 += len(m["stones"]) == 0
        rows.update(stones=len(st), cls=dict(collections.Counter(s["cls"] for s in st)),
                    mixed=float(np.mean([s["sec"] is not None for s in st])) if st else 0,
                    dmax_med=float(np.median([s["dmax"] for s in st])), small=float(np.mean([s["dmax"] < 4 for s in st])),
                    vol_med=float(np.median([s["vol"] for s in st])), vol_iqr=[float(x) for x in np.percentile([s["vol"] for s in st], [25, 75])],
                    loc=dict(collections.Counter(s["loc"] for s in st)), wed=[float(np.mean(wed)), float(np.std(wed))],
                    hydro=hyd / len(wed), calc=ncal / len(wed), stone_free=n0 / len(wed))
    out[sp] = rows
    print(sp, rows)
json.dump(out, open("/home/claude/results/cohort.json", "w"), indent=1)
