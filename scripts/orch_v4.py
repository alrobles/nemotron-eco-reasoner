#!/usr/bin/env python3
"""
Orchestrator v4 — mass GPU swarm with correct specs and --gres=gpu.

Key fixes over v3:
  - Correct GPU VRAM specs (A100=80 not 40, PRO6000=48 not 96)
  - Uses --gres=gpu:1 instead of --nodelist (Slurm picks any free GPU)
  - V100/Q6000 excluded by default (too weak, slow down swarm)
  - Higher default target (60 jobs)
"""

import subprocess, json, re, time, os, sys, argparse
from datetime import datetime
from collections import defaultdict

SCRATCH = os.path.expanduser("~/scratch")
CLUSTER_SCRATCH = "/home/a474r867/scratch"
REPO = os.path.join(CLUSTER_SCRATCH, "nemotron-eco-reasoner")
LOG = os.path.join(SCRATCH, "orch_v4.log")
STATE_FILE = os.path.join(SCRATCH, "gpu_state.json")
TEMPLATE = os.path.join(REPO, "hpc/nem_unified.slurm")

SIF = {
    "cuda":     os.path.join(REPO, "nemotron-cuda.sif"),
    "blackwell": os.path.join(REPO, "nemotron-blackwell.sif"),
    "mi210":    os.path.join(REPO, "nemotron-rocm.sif"),
}

# CORRECTED GPU specs — verified from cluster sinfo
GPU_SPECS = {
    "a100":    {"vram": 80, "bf16": True,  "arch": "ampere",    "seq": 2048, "rank": 64, "min_mem": 48, "sif": "cuda"},
    "mi210":   {"vram": 68, "bf16": True,  "arch": "cdna2",     "seq": 2048, "rank": 32, "min_mem": 64, "sif": "mi210"},
    "a40":     {"vram": 48, "bf16": True,  "arch": "ampere",    "seq": 1024, "rank": 16, "min_mem": 48, "sif": "cuda"},
    "l40":     {"vram": 48, "bf16": True,  "arch": "ada",       "seq": 1024, "rank": 16, "min_mem": 48, "sif": "cuda"},
    "pro6000": {"vram": 48, "bf16": True,  "arch": "blackwell", "seq": 2048, "rank": 32, "min_mem": 48, "sif": "blackwell"},
    "q8000":   {"vram": 48, "bf16": False, "arch": "turing",    "seq": 512,  "rank": 8,  "min_mem": 32, "sif": "cuda"},
    "v100":    {"vram": 32, "bf16": False, "arch": "volta",     "seq": 256,  "rank": 8,  "min_mem": 32, "sif": "cuda"},
    "q6000":   {"vram": 24, "bf16": False, "arch": "turing",    "seq": 128,  "rank": 4,  "min_mem": 24, "sif": "cuda"},
}

oom_count = defaultdict(int)
oom_until = {}

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

SSH_CMD = "ssh -i ~/.ssh/hpc_a474r867_ed25519_new -o BatchMode=yes -o ConnectTimeout=10 a474r867@hpc.crc.ku.edu"

def run(cmd, timeout=15):
    try:
        wrapped = f'{SSH_CMD} "{cmd}"'
        r = subprocess.run(wrapped, shell=True, capture_output=True, text=True, timeout=timeout)
        out = r.stdout.strip()
        if "Access to electronic resources" in out:
            idx = out.rfind("log out now.\n")
            if idx >= 0:
                out = out[idx + len("log out now.\n"):].strip()
        return out
    except Exception:
        return ""

def scan_nodes():
    """Get idle GPU nodes grouped by GPU type."""
    out = run("sinfo -p sixhour -N -O 'nodelist:|,statelong:|,memory:|,gres:' --noheader 2>/dev/null")
    nodes = {}
    for line in out.split("\n"):
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        name, state, mem_mb, gres = parts[0], parts[1], parts[2], parts[3]
        if state not in ("idle", "mixed"):
            continue
        for g in gres.split(","):
            g = g.strip()
            if not g.startswith("gpu:") or "shard" in g:
                continue
            m = re.match(r"gpu:(\w+):\d+\(S:(.+?)\)", g)
            if not m:
                continue
            gtype = m.group(1)
            if gtype not in GPU_SPECS:
                continue
            slots_str = m.group(2)
            slots = 0
            for part in slots_str.split(","):
                part = part.strip()
                if "-" in part:
                    lo, hi = part.split("-")
                    slots += int(hi) - int(lo) + 1
                else:
                    slots += 1
            try:
                ram_gb = int(mem_mb) // 1024
            except ValueError:
                continue
            # Count running jobs on this node
            alloc_out = run(f"squeue -w {name} -t R -o '%m' --noheader 2>/dev/null")
            alloc_jobs = len([x for x in alloc_out.split("\n") if x.strip()]) if alloc_out else 0
            alloc_mem = 0
            for aline in alloc_out.split("\n"):
                aline = aline.strip().upper()
                if not aline:
                    continue
                try:
                    if aline.endswith("G"):
                        alloc_mem += int(aline[:-1])
                    elif aline.endswith("M"):
                        alloc_mem += int(aline[:-1]) // 1024
                except ValueError:
                    pass
            free_gpus = max(0, slots - alloc_jobs)
            free_mem = ram_gb - alloc_mem
            min_mem = GPU_SPECS[gtype]["min_mem"]
            if free_gpus > 0 and free_mem >= min_mem:
                nodes[name] = {
                    "gtype": gtype,
                    "ram_gb": ram_gb,
                    "free_mem": free_mem,
                    "total_gpus": slots,
                    "free_gpus": free_gpus,
                    "state": state,
                }
    return nodes

def get_our_jobs():
    out = run("squeue -u a474r867 -t R,PD -o '%i|%j|%T|%N' --noheader 2>/dev/null")
    jobs = {}
    for line in out.split("\n"):
        p = line.strip().split("|")
        if len(p) >= 4 and p[1].startswith("nem"):
            jobs[p[0]] = {"state": p[2], "node": p[3]}
    return jobs

def submit_job(gtype, mem_gb, dry_run=False):
    """Submit one job with --gres=gpu:1 (Slurm picks the GPU)."""
    spec = GPU_SPECS[gtype]
    sif_key = spec["sif"]
    sif_path = SIF[sif_key]

    cluster_check = run(f"test -f {sif_path} && echo YES")
    if "YES" not in cluster_check:
        log(f"SKIP {gtype}: SIF {sif_path} not found")
        return False

    mem = min(mem_gb, 120)
    mem = max(mem, spec["min_mem"])

    # Use --gres=gpu:1 so Slurm picks any free GPU, not a specific node
    cmd = (f'sbatch --job-name=nem --mem={mem}G --gres=gpu:1 '
           f'--export=ALL,CONTAINER_SIF={sif_path} '
           f'{TEMPLATE} 2>/dev/null')

    if dry_run:
        log(f"DRY: {cmd}")
        return True

    out = run(cmd)
    if "Submitted" in out:
        jid = re.search(r"(\d+)", out)
        jid_str = jid.group(1) if jid else "?"
        log(f"SUBMIT {gtype} ({spec['vram']}GB) mem={mem}G -> job {jid_str}")
        return True
    else:
        log(f"FAIL: {out[:80]}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-jobs", type=int, default=60)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--skip-weak", action="store_true", default=True,
                       help="Skip V100/Q6000 (too slow, bf16=false)")
    args = parser.parse_args()

    log(f"=== ORCHESTRATOR V4 (target={args.target_jobs}, interval={args.interval}s, skip_weak={args.skip_weak}) ===")

    while True:
        try:
            nodes = scan_nodes()
            jobs = get_our_jobs()
            running = sum(1 for j in jobs.values() if j["state"] == "RUNNING")
            pending = sum(1 for j in jobs.values() if j["state"] == "PENDING")
            total = running + pending
            shortfall = max(0, args.target_jobs - total)

            # Group free GPUs by type
            gpu_free = defaultdict(int)
            for n, info in nodes.items():
                gtype = info["gtype"]
                # Skip weak GPUs that drag down the swarm
                if args.skip_weak and gtype in ("q6000", "v100"):
                    continue
                gpu_free[gtype] += info["free_gpus"]

            if shortfall > 0 and gpu_free:
                submitted = 0
                # Submit from best to worst GPU type
                for gtype in sorted(gpu_free, key=lambda g: GPU_SPECS[g]["vram"], reverse=True):
                    available = gpu_free[gtype]
                    spec = GPU_SPECS[gtype]
                    to_submit = min(shortfall - submitted, available)
                    for _ in range(to_submit):
                        if submit_job(gtype, spec["min_mem"] * 2, args.dry_run):
                            submitted += 1
                    if submitted >= shortfall:
                        break

                parts = [f"{g}:{gpu_free[g]}free" for g in sorted(gpu_free)]
                log(f"+{submitted} jobs | R={running} P={pending} T={total+submitted} | FREE: {' '.join(parts)}")
            else:
                parts = [f"{g}:{n}free" for g, n in sorted(gpu_free.items())] if gpu_free else ["none"]
                log(f"OK: R={running} P={pending} T={total} shortfall={shortfall} | FREE: {' '.join(parts)}")

            if args.once:
                break
            time.sleep(args.interval)
        except KeyboardInterrupt:
            log("Stopped.")
            break
        except Exception as e:
            log(f"ERR: {e}")
            if args.once:
                break
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
