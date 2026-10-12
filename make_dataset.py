"""Generate the RenalStone-DT-Sim cohorts (patient-wise splits, external sites, OOD and unlabelled pools)."""
import os, json, sys
import numpy as np
from multiprocessing import Pool
import simulator as S

ROOT = os.environ.get("SDT_DATA", "/home/claude/data")


def plan():
    rng = np.random.default_rng(2026)
    jobs = []
    pid = 0
    for site in "ABC":
        for k in range(500):
            pid += 1
            seed = 100000 + pid
            jobs.append(dict(pid=pid, scan=0, site=site, seed=seed, cohort="dev"))
            if rng.random() < 0.15:
                jobs.append(dict(pid=pid, scan=1, site=site, seed=seed, cohort="dev", noise_seed=900000 + pid))
    for site in "DE":
        for k in range(300):
            pid += 1
            jobs.append(dict(pid=pid, scan=0, site=site, seed=100000 + pid, cohort="ext" + site))
    for k in range(160):
        pid += 1
        site = "ABCD"[k % 4]
        jobs.append(dict(pid=pid, scan=0, site=site, seed=100000 + pid, cohort="ood", ood=True,
                         n_stones=int(rng.integers(1, 3))))
    for k in range(900):
        pid += 1
        jobs.append(dict(pid=pid, scan=0, site="ABC"[k % 3], seed=100000 + pid, cohort="unl", labelled=False))
    # patient-wise split of the development cohort
    dev_p = sorted({j["pid"] for j in jobs if j["cohort"] == "dev"})
    perm = rng.permutation(dev_p)
    n = len(perm)
    sp = {}
    for i, p in enumerate(perm):
        sp[int(p)] = "train" if i < .6 * n else "val" if i < .7 * n else "cal" if i < .8 * n else "test"
    for j in jobs:
        j["split"] = sp.get(j["pid"], j["cohort"])
    return jobs


def run(j):
    fn = f"{ROOT}/cases/p{j['pid']:05d}_s{j['scan']}.npz"
    if os.path.exists(fn):
        return fn
    o = S.simulate_case(j["seed"], j["site"], n_stones=j.get("n_stones"), ood=j.get("ood", False),
                        labelled=j.get("labelled", True), noise_seed=j.get("noise_seed"))
    meta = dict(j, wed=o["wed"], phase=o["phase"], stones=o.get("stones", []), hydro=bool(o.get("hydro", False)),
                n_calc=int(o.get("n_calc", 0)))
    arr = {k: o[k] for k in ("img", "bg", "clean", "occ", "inst", "anat", "distract") if k in o}
    np.savez(fn, meta=json.dumps(meta, default=str), **arr)
    return fn


if __name__ == "__main__":
    os.makedirs(f"{ROOT}/cases", exist_ok=True)
    jobs = plan()
    json.dump(jobs, open(f"{ROOT}/jobs.json", "w"))
    print(len(jobs), flush=True)
    with Pool(2) as p:
        for i, _ in enumerate(p.imap_unordered(run, jobs, chunksize=4)):
            if i % 200 == 0:
                print(i, flush=True)
    print("done", flush=True)
