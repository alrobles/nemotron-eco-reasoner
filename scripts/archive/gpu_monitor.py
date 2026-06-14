#!/usr/bin/env python3
"""
GPU Cluster Monitor — kbs partition, indefinite runtime.
Tracks all GPU types, utilization, and feeds adaptive submitter.
"""
import subprocess, json, re, time, os
from datetime import datetime
from collections import defaultdict

LOG = os.path.expanduser("~/scratch/gpu_monitor.log")
STATE_FILE = os.path.expanduser("~/scratch/gpu_state.json")
INTERVAL = 60

# GPU specs: {type: {vram_gb, bf16, arch, max_seq, max_rank, min_mem_gb, template}}
GPU_SPECS = {
    "pro6000": {"vram": 96, "bf16": True,  "arch": "Blackwell", "max_seq": 4096, "max_rank": 64, "min_mem": 64, "template": "nem_pro6000.slurm"},
    "q6000":   {"vram": 24, "bf16": False, "arch": "Turing",    "max_seq": 64,   "max_rank": 8,  "min_mem": 62, "template": "nem_q6000.slurm"},
    "v100":    {"vram": 32, "bf16": False, "arch": "Volta",     "max_seq": 96,   "max_rank": 8,  "min_mem": 62, "template": "nem_v100.slurm"},
    "a100":    {"vram": 40, "bf16": True,  "arch": "Ampere",    "max_seq": 2048, "max_rank": 32, "min_mem": 64, "template": "nem_a100.slurm"},
    "a40":     {"vram": 48, "bf16": True,  "arch": "Ampere",    "max_seq": 2048, "max_rank": 32, "min_mem": 64, "template": "nem_a100.slurm"},
    "l40":     {"vram": 48, "bf16": True,  "arch": "Ada",       "max_seq": 2048, "max_rank": 32, "min_mem": 64, "template": "nem_a100.slurm"},
    "q8000":   {"vram": 48, "bf16": False, "arch": "Turing",    "max_seq": 128,  "max_rank": 16, "min_mem": 62, "template": "nem_q6000.slurm"},
}

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def run(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        return r.stdout
    except:
        return ""

def scan_cluster():
    """Full cluster scan: nodes, GPUs, state, free resources."""
    nodes = {}
    out = run("sinfo -p sixhour -N -o '%N|%T|%m|%G|%C' --noheader 2>/dev/null")
    
    for line in out.strip().split("\n"):
        parts = line.strip().split("|")
        if len(parts) < 5:
            continue
        name, state, mem_mb, gres, cpus = parts[0], parts[1], parts[2], parts[3], parts[4]
        ram_gb = int(int(mem_mb) / 1024) if mem_mb.isdigit() else 0
        
        for g in gres.split(","):
            g = g.strip()
            if not g.startswith("gpu:") or "shard" in g:
                continue
            m = re.match(r"gpu:(\w+):\d+\(S:(.+?)\)", g)
            if not m:
                continue
            gtype = m.group(1)
            if gtype == "mi210":
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
            
            # Count allocated GPUs on this node
            alloc = run(f"squeue -w {name} -t R -o '%D' --noheader 2>/dev/null")
            alloc_gpus = sum(int(x) for x in alloc.strip().split("\n") if x.strip().isdigit())
            
            # Get our jobs on this node
            our_jobs = run(f"squeue -w {name} -u a474r867 -t R -o '%i' --noheader 2>/dev/null")
            our_count = len([x for x in our_jobs.strip().split("\n") if x.strip()])
            
            free = max(0, slots - alloc_gpus)
            
            if gtype not in nodes:
                nodes[gtype] = {"total_phys": 0, "total_slots": 0, "free_slots": 0, 
                               "idle_nodes": 0, "mixed_nodes": 0, "nodes": [],
                               "our_jobs": 0, "total_ram": 0}
            
            nodes[gtype]["total_slots"] += slots
            nodes[gtype]["free_slots"] += free
            nodes[gtype]["our_jobs"] += our_count
            nodes[gtype]["total_ram"] += ram_gb
            if state == "idle" and free > 0:
                nodes[gtype]["idle_nodes"] += 1
            elif state == "mixed":
                nodes[gtype]["mixed_nodes"] += 1
    
    return nodes

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "gpus": state}, f, indent=2)

def main():
    log("=== GPU MONITOR STARTED ===")
    
    while True:
        try:
            state = scan_cluster()
            save_state(state)
            
            total_free = sum(s["free_slots"] for s in state.values())
            total_ours = sum(s["our_jobs"] for s in state.values())
            
            # Summary line
            parts = []
            for gtype in sorted(state.keys()):
                s = state[gtype]
                spec = GPU_SPECS.get(gtype, {})
                vram = spec.get("vram", "?")
                parts.append(f"{gtype}:{s['free_slots']}free/{s['total_slots']}slot({vram}GB)")
            
            log(f"TOTAL free={total_free} ours={total_ours} | " + " | ".join(parts))
            
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            log("Monitor stopped.")
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
