"""Loading of simulated examinations into network tensors."""
import json, glob, os
import numpy as np
import physics as P

ROOT = os.environ.get("SDT_DATA", "/home/claude/data")
SCALE = 400.0


def meta_channels(site):
    ch = P.PROTOCOLS[site]["ch"]
    lo = ch[0][0]
    hi = ch[1][0] if len(ch) > 1 else ch[0][0]
    return np.array([lo / 150.0, hi / 150.0, P.PROTOCOLS[site]["slice"] / 3.0, float(len(ch) == 2)], np.float32)


def two_ch(a):
    a = a.astype(np.float32)
    return a if a.shape[0] == 2 else np.concatenate([a, a], 0)


def net_input(img, site):
    x = np.clip(two_ch(img) / SCALE, -2.5, 8.0)
    m = meta_channels(site)
    return np.concatenate([x, np.broadcast_to(m[:, None, None, None], (4,) + x.shape[1:])], 0).astype(np.float32)


def jobs():
    return json.load(open(f"{ROOT}/jobs.json"))


def case_path(j):
    return f"{ROOT}/cases/p{j['pid']:05d}_s{j['scan']}.npz"


def load_case(j):
    d = np.load(case_path(j))
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def split_jobs(name):
    return [j for j in jobs() if j["split"] == name]


def load_compact(name):
    """memory-lean training store: 2 image channels (float16) + protocol code; labels kept compact."""
    js = split_jobs(name)
    X, MV, Y = [], [], []
    for j in js:
        c = load_case(j)
        xi = net_input(c["img"], j["site"])
        X.append(xi[:2].astype(np.float16)); MV.append(xi[2:, 0, 0, 0].copy())
        Y.append(dict(occ=c["occ"], anat=c["anat"], bg=np.clip(two_ch(c["bg"]) / SCALE, -2.5, 8.0).astype(np.float16)))
        del c, xi
    return js, np.stack(X), np.stack(MV), Y


def expand(x2, mv):
    return np.concatenate([x2.astype(np.float32), np.broadcast_to(mv.astype(np.float32)[:, None, None, None], (4,) + x2.shape[1:])])


def load_split(name, labelled=True, limit=None):
    js = split_jobs(name)
    if limit:
        js = js[:limit]
    X, Y = [], []
    for j in js:
        c = load_case(j)
        X.append(net_input(c["img"], j["site"]).astype(np.float16))
        if labelled:
            Y.append(dict(occ=c["occ"], anat=c["anat"], bg=np.clip(two_ch(c["bg"]) / SCALE, -2.5, 8.0).astype(np.float16),
                          inst=c["inst"]))
    return js, np.stack(X), Y
