"""Two-lane queue: lane T trains (SSL + backbones) sequentially; lane I runs inference + twin fitting
on every backbone that has finished training. One thread per lane keeps peak memory below 3 GB."""
import os, subprocess, time, sys
RUNS = "/home/claude/runs"
ORDER = [(v, s) for s in (1, 2, 3) for v in ("full", "mtunet", "agunet")] + \
        [(v, s) for s in (1, 2) for v in ("nossl", "noanat", "nobg")]
ENV = dict(os.environ, NT="1", OMP_NUM_THREADS="1")


def lane_train():
    log = open(f"{RUNS}/laneT.log", "a")
    for s in (1, 2, 3):
        if not os.path.exists(f"{RUNS}/ssl_s{s}.pt"):
            subprocess.run(f"python3 -c \"import torch;torch.set_num_threads(1);import train;train.ssl_pretrain({s})\"",
                           shell=True, env=ENV, stdout=log, stderr=log)
    for v, s in ORDER:
        if not os.path.exists(f"{RUNS}/{v}_s{s}.pt"):
            t = time.time()
            subprocess.run(f"python3 train.py {v} {s}", shell=True, env=ENV, stdout=log, stderr=log)
            print(f"TRAINED {v} {s} {time.time()-t:.0f}s", file=log, flush=True)


def lane_infer():
    log = open(f"{RUNS}/laneI.log", "a")
    while True:
        todo = [(v, s) for v, s in ORDER if not os.path.exists(f"{RUNS}/rec_{v}_s{s}.pkl")]
        if not todo:
            break
        ready = [(v, s) for v, s in todo if os.path.exists(f"{RUNS}/{v}_s{s}.pt")]
        if not ready:
            time.sleep(60); continue
        v, s = ready[0]
        t = time.time()
        subprocess.run(f"python3 -W ignore infer.py {v} {s}", shell=True, env=ENV, stdout=log, stderr=log)
        print(f"INFERRED {v} {s} {time.time()-t:.0f}s", file=log, flush=True)


if __name__ == "__main__":
    os.makedirs(RUNS, exist_ok=True)
    (lane_train if sys.argv[1] == "T" else lane_infer)()
