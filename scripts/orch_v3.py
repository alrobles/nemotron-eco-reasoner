#!/usr/bin/env python3
"""
Adaptive Swarm Orchestrator v3 — intelligent GPU targeting + OOM backoff.

Merges smart_swarm.py node-targeting with orch_v2.py continuous loop.
Scans cluster, targets specific nodes with appropriate --mem, tracks OOM,
and auto-selects SIF per GPU architecture.

Usage (on HPC login node):
    python3 orch_v3.py [--target-jobs 20] [--interval 90] [--dry-run]
    
Or via Hermes:
    nohup python3 /home/a474r867/scratch/nemotron-eco-reasoner/scripts/orch_v3.py &
"""
import subprocess, json, re, time, os, sys, argparse
from datetime import datetime
from collections import defaultdict

SCRATCH = os.path.expanduser("~/scratch")
# Cluster-side scratch path (different from local on reumanlab)
CLUSTER_SCRATCH = "/home/a474r867/scratch"
REPO = os.path.join(CLUSTER_SCRATCH, "nemotron-eco-reasoner")
LOG = os.path.join(SCRATCH, "orch_v3.log")
STATE_FILE = os.path.join(SCRATCH, "gpu_state.json")
TEMPLATE = os.path.join(REPO, "hpc/nem_unified.slurm")

# SIF containers per architecture
SIF = {
    "cuda":     os.path.join(REPO, "nemotron-cuda.sif"),
    "blackwell": os.path.join(REPO, "nemotron-blackwell.sif"),
    "mi210":    os.path.join(REPO, "nemotron-rocm.sif"),
}

# GPU specs: vram, bf16, arch, recommended seq/rank, min RAM for job
GPU_SPECS = {
    "pro6000": {"vram": 96, "bf16": True,  "arch": "blackwell", "seq": 2048, "rank": 32, "min_mem": 64, "sif": "blackwell"},
    "a100":    {"vram": 40, "bf16": True,  "arch": "ampere",    "seq": 1024, "rank": 16, "min_mem": 64, "sif": "cuda"},
    "a40":     {"vram": 48, "bf16": True,  "arch": "ampere",    "seq": 1024, "rank": 16, "min_mem": 64, "sif": "cuda"},
    "l40":     {"vram": 48, "bf16": True,  "arch": "ada",       "seq": 1024, "rank": 16, "min_mem": 64, "sif": "cuda"},
    "q8000":   {"vram": 48, "bf16": False, "arch": "turing",    "seq": 512,  "rank": 8,  "min_mem": 48, "sif": "cuda"},
    "q6000":   {"vram": 24, "bf16": False, "arch": "turing",    "seq": 128,  "rank": 4,  "min_mem": 48, "sif": "cuda"},
    "v100":    {"vram": 32, "bf16": False, "arch": "volta",     "seq": 128,  "rank": 4,  "min_mem": 48, "sif": "cuda"},
    "mi210":   {"vram": 68, "bf16": True,  "arch": "cdna2",     "seq": 1024, "rank": 16, "min_mem": 64, "sif": "mi210"},
}

# Track OOM per node
oom_count = defaultdict(int)
oom_until = {}  # node -> timestamp when retry allowed

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

SSH_CMD = "ssh -i ~/.ssh/hpc_a474r867_ed25519_new -o BatchMode=yes -o ConnectTimeout=10 a474r867@hpc.crc.ku.edu"

def run(cmd, timeout=15):
    """Run command on cluster via SSH, stripping the login banner."""
    try:
        wrapped = f'{SSH_CMD} "{cmd}"'
        r = subprocess.run(wrapped, shell=True, capture_output=True, text=True, timeout=timeout)
        out = r.stdout.strip()
        # Strip login banner (everything before and including the last dashed line)
        if "Access to electronic resources" in out:
            # Find the last occurrence of "log out now." and take everything after
            idx = out.rfind("log out now.\n")
            if idx >= 0:
                out = out[idx + len("log out now.\n"):].strip()
        return out
    except Exception:
        return ""

def scan_nodes():
    """Scan cluster for available GPU nodes with free slots."""
    out = run("sinfo -p sixhour -N -o '%N|%T|%m|%G' --noheader 2>/dev/null")
    nodes = {}
    for line in out.split("\n"):
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        name, state, mem_mb, gres = parts[0], parts[1], parts[2], parts[3]
        if state not in ("idle", "mixed"):
            continue

        # Parse GPU type and slot count
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

            # Count jobs already on this node
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

            if free_gpus > 0 and free_mem > GPU_SPECS[gtype]["min_mem"]:
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
    """Get our running + pending jobs."""
    out = run("squeue -u a474r867 -t R,PD -o '%i|%j|%T|%N' --noheader 2>/dev/null")
    jobs = {}
    for line in out.split("\n"):
        p = line.strip().split("|")
        if len(p) >= 4 and p[1].startswith("nem"):
            jobs[p[0]] = {"state": p[2], "node": p[3]}
    return jobs

def detect_oom():
    """Check recent logs for OOM kills, apply backoff."""
    out = run("grep -rl 'Killed\\|CUDA out of memory\\|OutOfMemoryError' "
              f"{CLUSTER_SCRATCH}/nem_unified_*.err {CLUSTER_SCRATCH}/nem_unified_*.out 2>/dev/null")
    now = time.time()
    new_oom = []
    for f in out.split("\n"):
        if not f.strip():
            continue
        jid_m = re.search(r"(\d{8,})", f)
        if not jid_m:
            continue
        jid = jid_m.group(1)
        # Find which node
        node_out = run(f"sacct -j {jid} -o NodeList --noheader 2>/dev/null")
        node = node_out.strip().split("\n")[0].strip() if node_out.strip() else ""
        if not node or node == "None assigned":
            continue
        if node in oom_until and oom_until[node] > now:
            continue  # already in backoff
        oom_count[node] += 1
        backoff = min(300 * (2 ** min(oom_count[node], 5)), 3600)
        oom_until[node] = now + backoff
        new_oom.append((node, backoff))
        log(f"OOM: {node} (count={oom_count[node]}) backoff {backoff}s")
    return new_oom

def submit_targeted(node, gtype, mem_gb, dry_run=False):
    """Submit a job targeted to a specific node with appropriate resources."""
    spec = GPU_SPECS[gtype]
    sif_key = spec["sif"]
    sif_path = SIF[sif_key]

    if not os.path.exists(sif_path):
        # Check on cluster via SSH
        cluster_check = run(f"test -f {sif_path} && echo YES")
        if "YES" not in cluster_check:
            log(f"SKIP {node}: SIF {sif_path} not found")
            return False

    # Check OOM backoff
    now = time.time()
    if node in oom_until and oom_until[node] > now:
        remaining = int(oom_until[node] - now)
        log(f"SKIP {node}: OOM backoff ({remaining}s left)")
        return False

    mem = min(mem_gb, 120)  # cap at 120G
    mem = max(mem, spec["min_mem"])

    cmd = (f"sbatch --job-name=nem --nodelist={node} --mem={mem}G "
           f"--export=ALL,CONTAINER_SIF={sif_path} "
           f"{TEMPLATE} 2>/dev/null")

    if dry_run:
        log(f"DRY: {cmd}")
        return True

    out = run(cmd)
    if "Submitted" in out:
        jid = re.search(r"(\d+)", out)
        jid_str = jid.group(1) if jid else "?"
        log(f"SUBMIT {node} ({gtype} {spec['vram']}GB) mem={mem}G -> job {jid_str}")
        return True
    else:
        log(f"FAIL submit to {node}: {out[:80]}")
        return False

def save_state(nodes, jobs):
    """Save cluster state for monitoring."""
    state = {
        "timestamp": datetime.now().isoformat(),
        "nodes": nodes,
        "jobs": {k: v for k, v in jobs.items()},
        "oom_nodes": {k: v for k, v in oom_until.items() if v > time.time()},
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Adaptive Swarm Orchestrator v3")
    parser.add_argument("--target-jobs", type=int, default=20, help="Target number of concurrent jobs")
    parser.add_argument("--interval", type=int, default=90, help="Seconds between scans")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually submit")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    log(f"=== ORCHESTRATOR V3 STARTED (target={args.target_jobs}, interval={args.interval}s) ===")

    while True:
        try:
            nodes = scan_nodes()
            jobs = get_our_jobs()
            running = sum(1 for j in jobs.values() if j["state"] == "RUNNING")
            pending = sum(1 for j in jobs.values() if j["state"] == "PENDING")
            total = running + pending

            detect_oom()

            shortfall = max(0, args.target_jobs - total)

            if shortfall > 0 and nodes:
                # Sort nodes by GPU value: higher VRAM first (maximize training quality)
                sorted_nodes = sorted(
                    nodes.items(),
                    key=lambda x: GPU_SPECS.get(x[1]["gtype"], {}).get("vram", 0),
                    reverse=True
                )

                submitted = 0
                for node_name, info in sorted_nodes:
                    if submitted >= shortfall:
                        break
                    gtype = info["gtype"]
                    free = info["free_gpus"]
                    mem_per = min(int(info["free_mem"] / max(free, 1)), 120)

                    for _ in range(free):
                        if submitted >= shortfall:
                            break
                        if submit_targeted(node_name, gtype, mem_per, args.dry_run):
                            submitted += 1

                # Summary
                gpu_summary = defaultdict(lambda: {"running": 0, "free": 0})
                for n, info in nodes.items():
                    gt = info["gtype"]
                    gpu_summary[gt]["free"] += info["free_gpus"]
                for j in jobs.values():
                    if j["state"] == "RUNNING" and j["node"]:
                        # Try to identify GPU type from node
                        for n, info in nodes.items():
                            if n == j["node"]:
                                gpu_summary[info["gtype"]]["running"] += 1

                parts = [f"{gt}:{s['running']}R/{s['free']}F" for gt, s in sorted(gpu_summary.items())]
                log(f"+{submitted} jobs | R={running} P={pending} total={total+submitted} | {' '.join(parts)}")
            else:
                log(f"OK: R={running} P={pending} total={total} shortfall={shortfall} nodes_avail={len(nodes)}")

            save_state(nodes, jobs)

            # Cleanup old logs (>8h, more generous than v2's 6h)
            run(f"find {CLUSTER_SCRATCH}/ -name 'nem_unified_*.out' -mmin +480 -delete 2>/dev/null")
            run(f"find {CLUSTER_SCRATCH}/ -name 'nem_unified_*.err' -mmin +480 -delete 2>/dev/null")

            if args.once:
                break

            time.sleep(args.interval)
        except KeyboardInterrupt:
            log("Stopped by user.")
            break
        except Exception as e:
            log(f"Error: {e}")
            if args.once:
                break
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
