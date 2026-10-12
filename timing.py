"""Per-examination timing on the evaluation machine (one thread): network forward, anatomy-rendered
counterfactual, and twin fit per stone."""
import time, json, numpy as np, torch
import data as Dt, twin as T
from model import SDTBackbone
torch.set_num_threads(1)
m = SDTBackbone(True, True); m.load_state_dict(torch.load("/home/claude/runs/full_s1.pt")); m.eval()
js = [j for j in Dt.split_jobs("test") if j["scan"] == 0][:40]
tn, tb, tt, ns = [], [], [], 0
for j in js:
    c = Dt.load_case(j); img = c["img"].astype(np.float32); C = img.shape[0]
    x = torch.tensor(Dt.net_input(c["img"], j["site"]))[None]
    t0 = time.time()
    with torch.no_grad():
        r = m(x)
    tn.append(time.time() - t0)
    occ = r["occ"][0].numpy(); pa = torch.softmax(r["anat"][0], 0).numpy()
    t0 = time.time(); bg = T.anat_background(img, pa, occ, j["site"]); tb.append(time.time() - t0)
    t0 = time.time(); res = T.twin_fit(img, bg, occ, None, j["site"], c["meta"]["wed"], C); dt = time.time() - t0
    if res:
        tt.append(dt / len(res)); ns += len(res)
out = dict(net=float(np.median(tn)), bg=float(np.median(tb)), twin=float(np.median(tt)), n_exams=len(js), n_stones=ns)
json.dump(out, open("/home/claude/results/timing.json", "w"))
print(out)
