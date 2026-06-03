#!/usr/bin/env python3
"""
Nemotron Project Monitor — tracks all nem-* training jobs on KU HPC.
Runs via cron on reumanlab, queries cluster via ssh kuhpc.

Output:
  monitor/status.md   — human-readable status report
  monitor/history.jsonl — job history (append-only)

Usage:
  python3 monitor/nemotron_monitor.py
  # Or via cron: */5 * * * * cd ~/work/Github/nemotron-eco-reasoner && python3 monitor/nemotron_monitor.py
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MONITOR_DIR = REPO_ROOT / "monitor"
STATUS_FILE = MONITOR_DIR / "status.md"
HISTORY_FILE = MONITOR_DIR / "history.jsonl"
SSH_CMD = "ssh -o BatchMode=yes -o ConnectTimeout=10 kuhpc"
SCRATCH = "/home/a474r867/scratch/nemotron-eco-reasoner"
OUTPUT_DIR = f"{SCRATCH}/outputs/m1_dual"


def ssh(cmd: str) -> str:
    """Run command on cluster, return stdout."""
    try:
        r = subprocess.run(
            f"{SSH_CMD} '{cmd}'",
            shell=True, capture_output=True, text=True, timeout=30
        )
        return r.stdout.strip()
    except Exception:
        return ""


def get_nemotron_jobs() -> list[dict]:
    """Get all nem-* jobs from squeue + sacct (last 24h)."""
    jobs = {}

    # Running/Pending jobs
    out = ssh("squeue --me -o '%i|%j|%T|%M|%N|%r' --noheader | grep -i nem")
    for line in out.split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid = parts[0].strip()
        jobs[jid] = {
            "job_id": jid,
            "name": parts[1].strip(),
            "state": parts[2].strip(),
            "elapsed": parts[3].strip(),
            "nodes": parts[4].strip(),
            "reason": parts[5].strip() if len(parts) > 5 else "",
        }

    # Recently completed/failed (sacct last 2 days)
    # Slurm date format: YYYY-MM-DD, simpler than shell date -d
    two_days_ago = datetime.now().strftime("%Y-%m-%d")
    out = ssh(
        f"sacct --me -S 2026-05-31 "
        "-o 'JobID,JobName,State,Elapsed,ExitCode,NodeList' --noheader --parsable2 | grep -i nem"
    )
    for line in out.split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid = parts[0].strip().split(".")[0]  # strip .batch/.extern suffix
        if jid in jobs:
            continue  # already have from squeue
        if jid.isdigit():
            jobs[jid] = {
                "job_id": jid,
                "name": parts[1].strip(),
                "state": parts[2].strip(),
                "elapsed": parts[3].strip(),
                "exit_code": parts[4].strip().split(":")[0] if len(parts) > 4 else "",
                "nodes": parts[5].strip() if len(parts) > 5 else "",
                "reason": "",
            }

    return sorted(jobs.values(), key=lambda j: j["job_id"], reverse=True)


def get_training_progress(job_id: str) -> dict | None:
    """Parse training output for step/loss/GPU info."""
    outfile = f"{OUTPUT_DIR}/train_{job_id}.out"
    out = ssh(f"tail -30 {outfile} 2>/dev/null")

    if not out:
        return None

    progress = {"gpu_detected": False, "steps": [], "last_step": None}

    # GPU detection
    if "ROCm: True" in out:
        progress["gpu_detected"] = True
    gpu_match = re.findall(r"GPU (\d+): (.+?) \((\d+) GB\)", out)
    if gpu_match:
        progress["gpus"] = [{"id": int(m[0]), "name": m[1], "vram_gb": int(m[2])} for m in gpu_match]

    # mamba-ssm
    if "mamba-ssm OK" in out:
        progress["mamba_ssm"] = True

    # Training step: "[HH:MM:SS] Step N | loss=X.XX | ..."
    step_matches = re.findall(
        r"(\d{2}:\d{2}:\d{2}).*?[Ss]tep\s+(\d+).*?loss[=:]?\s*(\d+\.?\d*)", out
    )
    for m in step_matches:
        progress["steps"].append({"time": m[0], "step": int(m[1]), "loss": float(m[2])})

    if progress["steps"]:
        progress["last_step"] = progress["steps"][-1]

    # Trainer progress: {'loss': X, 'grad_norm': Y, 'learning_rate': Z, 'epoch': W}
    trainer_match = re.findall(
        r"\'loss\':\s*([\d.]+).*?\'grad_norm\':\s*([\d.]+).*?\'learning_rate\':\s*([\de.-]+).*?\'epoch\':\s*([\d.]+)",
        out
    )
    if trainer_match:
        m = trainer_match[-1]
        progress["trainer"] = {
            "loss": float(m[0]),
            "grad_norm": float(m[1]),
            "lr": m[2],
            "epoch": float(m[3]),
        }

    # Checkpoint
    if re.search(r"Saving.*checkpoint", out):
        progress["saving_checkpoint"] = True
    if "DONE!" in out or "Training complete" in out:
        progress["training_complete"] = True
        # Look for adapter
        adapter = ssh(f"ls -lh {OUTPUT_DIR}/adapter_model.safetensors 2>/dev/null")
        if adapter:
            progress["adapter_saved"] = adapter.strip()

    return progress


def get_error(job_id: str) -> str | None:
    """Get last 10 lines of stderr."""
    errfile = f"{OUTPUT_DIR}/train_{job_id}.err"
    return ssh(f"tail -10 {errfile} 2>/dev/null")


def load_history() -> list[dict]:
    """Load existing history."""
    if not HISTORY_FILE.exists():
        return []
    history = []
    with open(HISTORY_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return history


def save_history(job: dict):
    """Append a job record to history."""
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(job) + "\n")


def update_history(jobs: list[dict]):
    """Update history with new/updated job statuses."""
    existing = load_history()
    existing_ids = {j["job_id"] for j in existing}

    for job in jobs:
        jid = job["job_id"]
        record = {
            "job_id": jid,
            "name": job.get("name", ""),
            "state": job.get("state", ""),
            "elapsed": job.get("elapsed", ""),
            "nodes": job.get("nodes", ""),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add exit code for terminal states
        if job.get("state") in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
            if "exit_code" in job:
                record["exit_code"] = job["exit_code"]
            else:
                # Try sacct for exit code
                out = ssh(f"sacct -j {jid} -o ExitCode --noheader --parsable2 2>/dev/null | head -1")
                if out:
                    record["exit_code"] = out.split(":")[0]

        # Update if exists, append if new
        if jid in existing_ids:
            # Update last record only if state changed
            for i, e in enumerate(existing):
                if e["job_id"] == jid and e.get("state") != record.get("state"):
                    existing[i] = record
        else:
            existing.append(record)

    # Rewrite history
    with open(HISTORY_FILE, "w") as f:
        for e in existing:
            f.write(json.dumps(e) + "\n")


def generate_status(jobs: list[dict]) -> str:
    """Generate status.md content."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"# Nemotron Training Status — {now}",
        "",
    ]

    # Active jobs
    active = [j for j in jobs if j.get("state") in ("RUNNING", "PENDING")]
    if active:
        lines.append("## Active Jobs")
        lines.append("")
        for j in active:
            state_icon = "🟢" if j["state"] == "RUNNING" else "🟡"
            lines.append(
                f"| {state_icon} **{j['job_id']}** | {j.get('name','')} | "
                f"{j['state']} | {j.get('elapsed','')} | {j.get('nodes','')} |"
            )
            lines.append("|---|---|---|---|---|")

            # Training progress
            if j["state"] == "RUNNING":
                prog = get_training_progress(j["job_id"])
                if prog:
                    if prog.get("gpu_detected"):
                        gpus = prog.get("gpus", [])
                        gpu_str = ", ".join(
                            f"GPU {g['id']}: {g['name']} ({g['vram_gb']}GB)" for g in gpus
                        )
                        lines.append(f"**GPUs:** {gpu_str}  ")
                    if prog.get("mamba_ssm"):
                        lines.append("**mamba-ssm:** ✅ Installed  ")
                    if prog.get("last_step"):
                        s = prog["last_step"]
                        lines.append(f"**Progress:** Step {s['step']}, loss={s['loss']:.4f}  ")
                    if prog.get("trainer"):
                        t = prog["trainer"]
                        lines.append(
                            f"**Trainer:** loss={t['loss']:.4f}, "
                            f"grad_norm={t['grad_norm']:.4f}, "
                            f"lr={t['lr']}, epoch={t['epoch']:.2f}  "
                        )
                    if prog.get("saving_checkpoint"):
                        lines.append("**Checkpoint:** Saving...  ")
                    if prog.get("training_complete"):
                        lines.append("**✅ Training complete!**  ")
                        if prog.get("adapter_saved"):
                            lines.append(f"**Adapter:** {prog['adapter_saved']}  ")
                    if not prog.get("gpu_detected") and not prog.get("steps"):
                        lines.append("⏳ Container loading...  ")
            lines.append("")
    else:
        lines.append("## No Active Jobs")
        lines.append("")

    # Recent history
    terminal = [j for j in jobs if j.get("state") in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT")]
    if terminal:
        lines.append("## Recent History (last 48h)")
        lines.append("")
        lines.append("| Job ID | Name | State | Elapsed | Exit | Nodes |")
        lines.append("|--------|------|-------|---------|------|-------|")
        for j in terminal[:15]:
            state_icon = {"COMPLETED": "✅", "FAILED": "❌", "CANCELLED": "🚫", "TIMEOUT": "⏰"}.get(
                j["state"], "❓"
            )
            lines.append(
                f"| {state_icon} {j['job_id']} | {j.get('name','')} | {j['state']} | "
                f"{j.get('elapsed','')} | {j.get('exit_code','')} | {j.get('nodes','')} |"
            )
        lines.append("")

    # Errors
    failed = [j for j in jobs if j.get("state") == "FAILED"]
    if failed:
        lines.append("## Last Errors")
        lines.append("")
        for j in failed[:3]:
            err = get_error(j["job_id"])
            if err:
                lines.append(f"### Job {j['job_id']}")
                lines.append("```")
                lines.append(err)
                lines.append("```")
                lines.append("")

    lines.append(f"*Auto-generated by nemotron_monitor.py — {now}*")
    return "\n".join(lines)


def main():
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)

    jobs = get_nemotron_jobs()
    update_history(jobs)

    status = generate_status(jobs)
    with open(STATUS_FILE, "w") as f:
        f.write(status)

    # Quick summary to stdout
    active = [j for j in jobs if j["state"] in ("RUNNING", "PENDING")]
    terminal = [j for j in jobs if j["state"] in ("COMPLETED", "FAILED")]
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"nemotron: {len(active)} active, {len(terminal)} terminal "
        f"-> {STATUS_FILE}"
    )


if __name__ == "__main__":
    main()
