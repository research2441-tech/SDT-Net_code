"""Variance-model parameters of the fused volume (validation split), averaged over SDT-Net seeds."""
import json, numpy as np, evaluate as E
cs = [E.fit_vol_model(E.load("full", s)) for s in (1, 2, 3)]
out = {k: float(np.mean([c[k] for c in cs])) for k in cs[0]}
out["per_seed"] = cs
json.dump(out, open("/home/claude/results/volcal.json", "w"), indent=1)
print(out)
