"""Second inference lane for ablation and sensitivity runs; starts once backbone training has finished."""
import os, subprocess, time
RUNS = "/home/claude/runs"
ENV = dict(os.environ, NT="1", OMP_NUM_THREADS="1")
EV = "--splits test,extD,extE"
JOBS = [("full", 1, "--bg net", "_bgnet"), ("full", 1, "--bg annulus", "_bgann"),
        ("full", 1, f"--psf 0.8 {EV}", "_psf080"), ("full", 1, f"--psf 1.25 {EV}", "_psf125"),
        ("full", 2, "--bg net", "_bgnet"), ("full", 2, "--bg annulus", "_bgann")]
log = open(f"{RUNS}/lane3.log", "a")
while not os.path.exists(f"{RUNS}/nobg_s2.pt"):
    time.sleep(120)
for v, s, args, tag in JOBS:
    if os.path.exists(f"{RUNS}/rec_{v}_s{s}{tag}.pkl"):
        continue
    t = time.time()
    subprocess.run(f"python3 -W ignore infer.py {v} {s} {args} --tag {tag}", shell=True, env=ENV, stdout=log, stderr=log)
    print(f"INFERRED {v} {s} {tag} {time.time()-t:.0f}s", file=log, flush=True)
