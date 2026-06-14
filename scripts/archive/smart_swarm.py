#!/usr/bin/env python3
"""
Smart GPU swarm submitter for Nemotron training.
Queries cluster state, calculates per-node capacity, submits optimally.
"""
import subprocess, json, re, sys

SSH = ["ssh", "kuhpc"]
TEMPLATE_Q6000 = "/home/a474r867/scratch/nem_q6k_v7.slurm"
TEMPLATE_ANY = "/home/a474r867/scratch/nem_any_v7.slurm"
MIN_MEM_PER_JOB = 48  # GB — minimum for Nemotron 30B 4-bit loading
MAX_MEM_PER_JOB = 64  # GB — ceiling

def ssh(cmd):
    result = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=20)
    return result.stdout

def get_nodes():
    """Get all GPU nodes with state and resources."""
    out = ssh("sinfo -p sixhour -N -o '%N|%T|%m|%G' --noheader 2>/dev/null")
    nodes = {}
    for line in out.split("\n"):
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        name, state, mem_mb, gres = parts[0], parts[1], parts[2], parts[3]
        if state not in ("idle", "mixed"):
            continue
        if "q6000" not in gres and "v100" not in gres and "pro6000" not in gres and "a100" not in gres and "a40" not in gres and "l40" not in gres and "q8000" not in gres:
            continue
        try:
            mem_gb = int(mem_mb) / 1024
        except:
            continue
        # Count allocatable GPUs from GRES
        gpu_count = 0
        for g in gres.split(","):
            g = g.strip()
            if not g.startswith("gpu:"):
                continue
            if "mi210" in g or "shard" in g:
                continue
            m = re.search(r'\(S:(.+?)\)', g)
            if m:
                slot_str = m.group(1)
                # Parse slots: "0-1" = 2, "0-1,3" = 3, "0" = 1
                slots = 0
                for part in slot_str.split(","):
                    part = part.strip()
                    if "-" in part:
                        lo, hi = part.split("-")
                        slots += int(hi) - int(lo) + 1
                    else:
                        slots += 1
                gpu_count += slots
        if gpu_count == 0:
            continue
        
        # Check allocated jobs on this node
        alloc_out = ssh(f"squeue -w {name} -t R -o '%m' --noheader 2>/dev/null")
        alloc_mem = 0
        for aline in alloc_out.split("\n"):
            aline = aline.strip().upper()
            if not aline:
                continue
            if aline.endswith("G"):
                alloc_mem += int(aline[:-1].rstrip("M").rstrip("G")) if "M" not in aline else 0
            elif aline.endswith("M"):
                alloc_mem += int(aline[:-1]) / 1024
        alloc_gpus = len(alloc_out.strip().split("\n")) if alloc_out.strip() else 0
        free_mem = mem_gb - alloc_mem
        free_gpus = gpu_count - alloc_gpus

        if free_gpus > 0 and free_mem > MIN_MEM_PER_JOB:
            nodes[name] = {
                "mem_gb": mem_gb, "free_mem": free_mem,
                "gpu_count": gpu_count, "free_gpus": free_gpus,
                "gres": gres, "state": state
            }
    return nodes

def main():
    nodes = get_nodes()
    total_jobs = 0
    
    print(f"=== SMART SWARM SUBMITTER ===")
    print(f"{'Node':<15} {'RAM':>8} {'Free':>8} {'GPUs':>6} {'Free':>6} {'Jobs':>6} {'Mem/job':>8}")
    print("-" * 65)
    
    for name, info in sorted(nodes.items()):
        free_mem = info["free_mem"]
        free_gpus = info["free_gpus"]
        max_by_gpu = int(free_gpus)
        
        # Max jobs: use ALL free GPUs, limited by RAM
        # Try to fit as many as possible, stepping down --mem if needed
        n_jobs = 0
        mem_per_job = MAX_MEM_PER_JOB
        while mem_per_job >= MIN_MEM_PER_JOB and n_jobs < max_by_gpu:
            n_jobs = min(max_by_gpu, int(free_mem / mem_per_job))
            if n_jobs > 0:
                break
            mem_per_job -= 8  # step down by 8GB
        
        if n_jobs == 0:
            continue
        
        # Use all free GPUs if RAM allows
        if n_jobs < max_by_gpu and free_mem / n_jobs >= MIN_MEM_PER_JOB:
            # Can we fit more?
            extra = min(max_by_gpu - n_jobs, int((free_mem - n_jobs * mem_per_job) / MIN_MEM_PER_JOB))
            n_jobs += extra
        
        mem_per_job = min(int(free_mem / n_jobs), MAX_MEM_PER_JOB)
        mem_per_job = max(mem_per_job, MIN_MEM_PER_JOB)
        
        # Choose template
        if "q6000" in info["gres"]:
            template = TEMPLATE_Q6000
            jobname = "nq7"
        else:
            template = TEMPLATE_ANY
            jobname = "nv7"
        
        print(f"{name:<15} {info['mem_gb']:>7.0f}G {free_mem:>7.0f}G {info['gpu_count']:>5} {free_gpus:>5} {n_jobs:>5} {mem_per_job:>7}G")
        
        # Submit jobs
        for _ in range(n_jobs):
            cmd = f"sbatch --job-name={jobname} --nodelist={name} --mem={mem_per_job}G {template}"
            out = ssh(cmd)
            if "Submitted" in out:
                total_jobs += 1
    
    print("-" * 65)
    print(f"TOTAL: {total_jobs} jobs submitted across {len([n for n in nodes if nodes[n]['free_gpus'] > 0])} nodes")
    return total_jobs

if __name__ == "__main__":
    main()
