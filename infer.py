"""Inference of a trained backbone + spectral twin on every evaluation cohort.
Writes per-scan and per-instance records used by heads.py."""
import os, sys, time, pickle, argparse
import numpy as np
import torch
from scipy import ndimage as ndi
import data as Dt
import physics as P
import twin as T
from model import SDTBackbone
from train import VARIANTS, RUNS

SPLITS = [("train", 600), ("val", None), ("cal", None), ("test", None), ("extD", None), ("extE", None), ("ood", None)]


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9)


def hu_stats(img, K):
    v = img[:, K].astype(np.float64)
    mean, mx = v.mean(1), v.max(1)
    der = mean[0] / max(mean[-1], 1.0) if img.shape[0] == 2 else 0.0
    return dict(hu_mean=mean, hu_max=mx, der=der)


def run(variant, seed, splits=SPLITS, refine=True, psf_scale=1.0, tag="", bg_choice="auto"):
    cfg = VARIANTS[variant]
    m = SDTBackbone(cfg["anat"], cfg["bg"])
    m.load_state_dict(torch.load(f"{RUNS}/{variant}_s{seed}.pt"))
    m.eval()
    out = dict(scans=[], inst=[])
    t0 = time.time()
    for split, lim in splits:
        js = Dt.split_jobs(split)
        if lim:
            js = [js[i] for i in np.unique(np.linspace(0, len(js) - 1, lim).astype(int))]   # spread over all sites
        for j in js:
            c = Dt.load_case(j)
            site = j["site"]
            C = len(P.PROTOCOLS[site]["ch"])
            x = torch.tensor(Dt.net_input(c["img"], site))[None]
            with torch.no_grad():
                r = m(x)
            occ = r["occ"][0].numpy()
            feat = r["feat"][0].numpy()
            img = c["img"].astype(np.float32)
            choice = bg_choice
            if choice == "auto":
                choice = "anat" if cfg["anat"] else "annulus"
            if choice == "anat":
                pa = torch.softmax(r["anat"][0], 0).numpy()
                bg = T.anat_background(img, pa, occ, site)
                bmode = "net"                      # use the supplied background field
            elif choice == "net":
                bg = r["bg"][0].numpy() * Dt.SCALE
                bmode = "net"
            else:
                bg = None
                bmode = "annulus"
            netbg = r["bg"][0].numpy() * Dt.SCALE if cfg["bg"] else None
            meta = c["meta"]
            gt = c["inst"]
            srec = dict(split=split, pid=j["pid"], scan=j["scan"], site=site, wed=meta["wed"], n_gt=len(meta["stones"]),
                        gt_vol=[s["vol"] for s in meta["stones"]], gt_dmax=[s["dmax"] for s in meta["stones"]])
            if cfg["anat"]:
                pa = r["anat"][0].argmax(0).numpy()
                srec["dice_cs"] = dice(pa == 2, c["anat"] == 2)
                srec["dice_par"] = dice(pa == 1, c["anat"] == 1)
            near = ndi.binary_dilation(c["occ"] > 0.01, iterations=3)
            if near.any() and bg is not None:
                srec["bg_mae"] = float(np.abs(bg[:C] - c["bg"][:C].astype(np.float32))[:, near].mean())
            if near.any() and netbg is not None:
                srec["bg_mae_net"] = float(np.abs(netbg[:C] - c["bg"][:C].astype(np.float32))[:, near].mean())
            gt_occ = c["occ"].astype(np.float32)
            gt_bin = gt_occ > 0.5
            srec["dice_stone"] = dice(occ > 0.5, gt_bin) if gt_bin.any() else np.nan
            res = T.twin_fit(img, bg, occ, feat, site, meta["wed"], C, refine=refine, bg_mode=bmode,
                             psf_scale=psf_scale)
            matched = {}
            for q in res:
                K = q.pop("mask_k")
                ov = np.bincount(gt[K], minlength=len(meta["stones"]) + 1)[1:]
                q.update(hu_stats(img, K))
                q.update(split=split, pid=j["pid"], scan=j["scan"], site=site, n_ch=C, wed=meta["wed"],
                         on_distractor=bool(c["distract"][K].any()))
                if ov.sum() > 0:
                    gi = int(ov.argmax())
                    s = meta["stones"][gi]
                    q.update(gt_id=gi + 1, cls=s["cls"], gt_vol=s["vol"], gt_deq=s["deq"], gt_dmax=s["dmax"],
                             loc=s["loc"], mixed=s["sec"] is not None, mode=s["mode"], ood=s["ood"],
                             pack=s["pack"], overlap=int(ov[gi]))
                else:
                    q.update(gt_id=0)
                out["inst"].append(q)
            # resolve duplicates: one TP per GT stone (largest overlap)
            mine = [q for q in out["inst"] if q["pid"] == j["pid"] and q["scan"] == j["scan"] and q["split"] == split]
            best = {}
            for q in mine:
                if q["gt_id"] > 0 and (q["gt_id"] not in best or q["overlap"] > best[q["gt_id"]]["overlap"]):
                    best[q["gt_id"]] = q
            for q in mine:
                q["tp"] = q["gt_id"] > 0 and best[q["gt_id"]] is q
            srec["n_tp"] = len(best)
            srec["n_fp"] = sum(1 for q in mine if not q["tp"])
            srec["n_fp_distractor"] = sum(1 for q in mine if not q["tp"] and q["on_distractor"])
            srec["detected_ids"] = sorted(best)
            out["scans"].append(srec)
        print(f"{variant} s{seed} {split} done {time.time()-t0:.0f}s", flush=True)
    fn = f"{RUNS}/rec_{variant}_s{seed}{tag}.pkl"
    pickle.dump(out, open(fn, "wb"))
    return fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant"); ap.add_argument("seed", type=int)
    ap.add_argument("--norefine", action="store_true")
    ap.add_argument("--psf", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--splits", default="")
    ap.add_argument("--bg", default="auto")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("NT", "2")))
    sp = SPLITS if not a.splits else [(s, None) for s in a.splits.split(",")]
    run(a.variant, a.seed, sp, refine=not a.norefine, psf_scale=a.psf, tag=a.tag, bg_choice=a.bg)
