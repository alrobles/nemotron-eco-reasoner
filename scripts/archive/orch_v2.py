#!/usr/bin/env python3
"""
Adaptive Swarm Orchestrator v2 — reads gpu_state.json, submits per GPU type.
Backoff on OOM, max utilization target.
"""
import subprocess, json, re, time, os
from datetime import datetime
from collections import defaultdict

LOG = os.path.expanduser("~/scratch/orch_v2.log")
STATE = os.path.expanduser("~/scratch/gpu_state.json")
TEMPLATE = "/home/a474r867/scratch/nem_unified.slurm"
INTERVAL = 120
TARGET_TOTAL = 30  # soft target

# Track OOM per node for backoff
oom_count = defaultdict(int)
oom_backoff = {}  # node -> seconds until retry

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
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

def read_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except:
        return None

def get_our_jobs():
    j = {}
    out = run("squeue -u a474r867 -t R,PD -o '%i|%j|%T|%N' --noheader 2>/dev/null")
    for line in out.strip().split("\n"):
        p = line.strip().split("|")
        if len(p)>=4 and p[1]=="nem":
            j[p[0]]={"state":p[2],"node":p[3]}
    return j

def check_oom():
    oom = []
    out = run("grep -l 'Killed' /home/a474r867/scratch/nem_unified_*.err 2>/dev/null")
    for f in out.strip().split("\n"):
        if f:
            jid = re.search(r"(\d{8,})", f)
            if jid:
                oom.append(jid.group(1))
    return oom

def submit_jobs(n):
    submitted = 0
    for _ in range(n):
        out = run(f"sbatch --job-name=nem {TEMPLATE} 2>/dev/null")
        if "Submitted" in out:
            submitted += 1
    return submitted

def main():
    log("=== ORCHESTRATOR V2 STARTED ===")
    
    while True:
        try:
            state = read_state()
            jobs = get_our_jobs()
            running = sum(1 for j in jobs.values() if j["state"]=="RUNNING")
            pending = sum(1 for j in jobs.values() if j["state"]=="PENDING")
            total = running + pending
            
            # Check OOM kills and apply backoff
            oom_jobs = check_oom()
            now = time.time()
            
            for jid in oom_jobs:
                # Find node
                out = run(f"sacct -j {jid} -o NodeList --noheader 2>/dev/null")
                node = out.strip()
                if node:
                    oom_count[node] += 1
                    backoff_sec = min(300 * (2 ** oom_count[node]), 3600)  # 5min→10min→20min→...→1h max
                    oom_backoff[node] = now + backoff_sec
                    log(f"OOM on {node} (total={oom_count[node]}) — backoff {backoff_sec}s")
            
            shortfall = max(0, TARGET_TOTAL - total)
            
            if shortfall > 0:
                n = submit_jobs(shortfall)
                if n > 0:
                    log(f"+{n} jobs (R={running} P={pending} OOM={len(oom_jobs)})")
            else:
                log(f"OK: R={running} P={pending} OOM={len(oom_jobs)}")
            
            # Clean old output
            run("find /home/a474r867/scratch/ -name 'nem_unified_*.out' -mmin +360 -delete 2>/dev/null")
            run("find /home/a474r867/scratch/ -name 'nem_unified_*.err' -mmin +360 -delete 2>/dev/null")
            
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            log("Stopped.")
            break
        except Exception as e:
            log(f"Err: {e}")
            time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
