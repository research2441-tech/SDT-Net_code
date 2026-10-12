"""Extra inference lane over the main queue in reverse order (after training ends); skips jobs that another
lane is currently running."""
import os, subprocess, time
from runner2 import ORDER, RUNS, ENV
log = open(f"{RUNS}/lane4.log", "a")
while not os.path.exists(f"{RUNS}/nobg_s2.pt"):
    time.sleep(120)
while True:
    running = subprocess.run("ps aux", shell=True, capture_output=True, text=True).stdout
    todo = [(v, s) for v, s in reversed(ORDER) if not os.path.exists(f"{RUNS}/rec_{v}_s{s}.pkl")
            and f"infer.py {v} {s}" not in running]
    if not todo:
        break
    v, s = todo[0]
    t = time.time()
    subprocess.run(f"python3 -W ignore infer.py {v} {s}", shell=True, env=ENV, stdout=log, stderr=log)
    print(f"INFERRED {v} {s} {time.time()-t:.0f}s", file=log, flush=True)
