"""Classical pipelines: fixed Hounsfield threshold detection with (i) LDA on attenuation and
dual-energy-ratio descriptors and (ii) random forest on radiomic descriptors."""
import os, pickle, time
import numpy as np
from scipy import ndimage as ndi
import data as Dt
import physics as P
from infer import SPLITS
from train import RUNS

THR = 200.0


def radiomics(img, K):
    v = img[:, K].astype(np.float64)
    C = img.shape[0]
    f = []
    for c in range(2):
        x = v[min(c, C - 1)]
        f += [x.mean(), x.max(), x.std(), *np.percentile(x, [10, 50, 90])]
    pts = np.argwhere(K)
    ext = pts.max(0) - pts.min(0) + 1
    n = K.sum()
    surf = n - ndi.binary_erosion(K).sum()
    gm = np.sqrt(sum(g ** 2 for g in np.gradient(img[-1].astype(np.float64))))[K].mean()
    der = v[0].mean() / max(v[-1].mean(), 1.0) if C == 2 else 0.0
    return np.array(f + [np.log(n), *np.sort(ext), surf / n, gm, der, float(C == 2)])


def run():
    out = dict(scans=[], inst=[])
    for split, lim in SPLITS:
        js = Dt.split_jobs(split)
        if lim:
            js = [js[i] for i in np.unique(np.linspace(0, len(js) - 1, lim).astype(int))]   # spread over all sites
        for j in js:
            c = Dt.load_case(j)
            img = c["img"].astype(np.float32)
            lab, n = ndi.label(img.max(0) > THR, structure=np.ones((3, 3, 3)))
            gt, meta = c["inst"], c["meta"]
            mine = []
            for k in range(1, n + 1):
                K = lab == k
                if K.sum() < 2:
                    continue
                ov = np.bincount(gt[K], minlength=len(meta["stones"]) + 1)[1:]
                q = dict(split=split, pid=j["pid"], scan=j["scan"], site=j["site"], n_ch=img.shape[0],
                         feat=radiomics(img, K), V_thr=float(K.sum()), on_distractor=bool(c["distract"][K].any()),
                         dmax=float((np.ptp(np.argwhere(K), 0) + 1).max()))
                if ov.sum() > 0:
                    gi = int(ov.argmax()); s = meta["stones"][gi]
                    q.update(gt_id=gi + 1, cls=s["cls"], gt_vol=s["vol"], gt_deq=s["deq"], gt_dmax=s["dmax"],
                             loc=s["loc"], mixed=s["sec"] is not None, ood=s["ood"], overlap=int(ov[gi]))
                else:
                    q["gt_id"] = 0
                mine.append(q)
            best = {}
            for q in mine:
                if q["gt_id"] > 0 and (q["gt_id"] not in best or q["overlap"] > best[q["gt_id"]]["overlap"]):
                    best[q["gt_id"]] = q
            for q in mine:
                q["tp"] = q["gt_id"] > 0 and best[q["gt_id"]] is q
            out["inst"] += mine
            out["scans"].append(dict(split=split, pid=j["pid"], scan=j["scan"], site=j["site"], n_gt=len(meta["stones"]),
                                     gt_vol=[s["vol"] for s in meta["stones"]], gt_dmax=[s["dmax"] for s in meta["stones"]],
                                     n_tp=len(best), n_fp=sum(1 for q in mine if not q["tp"]),
                                     n_fp_distractor=sum(1 for q in mine if not q["tp"] and q["on_distractor"]),
                                     detected_ids=sorted(best)))
        print("classic", split, flush=True)
    pickle.dump(out, open(f"{RUNS}/rec_classic.pkl", "wb"))


if __name__ == "__main__":
    os.makedirs(RUNS, exist_ok=True)
    run()
