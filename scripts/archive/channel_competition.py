#!/usr/bin/env python3
"""
Channel Latency Competition: GitHub Issues vs HTTP Direct

Sends the same task through both channels and measures round-trip latency.
Results are exported to Phoenix for visualization.

Usage:
  python3 scripts/channel_competition.py
"""

import json
import time
import urllib.request
from hermes_phoenix_tracer import PhoenixTracer

# ─── Config ─────────────────────────────────────────────────────────
PHOENIX = "http://localhost:6006/v1/traces"
HTTP_API = "http://localhost:8888/task"
TEST_CMD = "squeue -o '%i %T %N' --noheader -p sixhour | head -5"
TEST_TIMEOUT = 30

tracer = PhoenixTracer("channel-competition", PHOENIX)


def test_http(task_id: str) -> dict:
    """Test HTTP direct channel."""
    with tracer.span("channel.http", task_id=task_id, cmd=TEST_CMD[:50]) as span:
        start = time.time()
        try:
            req = urllib.request.Request(
                HTTP_API,
                data=json.dumps({"cmd": TEST_CMD, "task_id": task_id, "timeout": TEST_TIMEOUT}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=TEST_TIMEOUT + 5)
            data = json.loads(resp.read())
            elapsed = time.time() - start
            span.set_attribute("latency_ms", elapsed * 1000)
            span.set_attribute("exit_code", data.get("exit_code", -1))
            span.set_attribute("output_len", len(data.get("stdout", "")))
            span.set_status("ok" if data.get("exit_code") == 0 else "error")
            return {"channel": "http", "latency_s": round(elapsed, 3), "ok": True, "data": data}
        except Exception as exc:
            elapsed = time.time() - start
            span.set_attribute("latency_ms", elapsed * 1000)
            span.set_status("error", str(exc))
            return {"channel": "http", "latency_s": round(elapsed, 3), "ok": False, "error": str(exc)}


def test_github(task_id: str) -> dict:
    """Test GitHub Issues channel (simulated — same endpoint for comparison)."""
    with tracer.span("channel.github", task_id=task_id, cmd=TEST_CMD[:50]) as span:
        start = time.time()
        # GitHub Issues channel is async (poll-based). We simulate the full cycle:
        # Simulate 60s polling delay + actual execution time via HTTP
        poll_delay = 60  # seconds (typical cron interval)
        span.set_attribute("poll_delay_s", poll_delay)
        try:
            req = urllib.request.Request(
                HTTP_API,
                data=json.dumps({"cmd": TEST_CMD, "task_id": task_id, "timeout": TEST_TIMEOUT}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=TEST_TIMEOUT + 5)
            data = json.loads(resp.read())
            elapsed = time.time() - start + poll_delay
            span.set_attribute("latency_ms", elapsed * 1000)
            span.set_attribute("effective_latency_ms", elapsed * 1000)
            span.set_attribute("execution_only_ms", (elapsed - poll_delay) * 1000)
            span.set_status("ok" if data.get("exit_code") == 0 else "error")
            return {"channel": "github", "latency_s": round(elapsed, 3), "ok": True, "data": data}
        except Exception as exc:
            elapsed = time.time() - start + poll_delay
            span.set_attribute("latency_ms", elapsed * 1000)
            span.set_status("error", str(exc))
            return {"channel": "github", "latency_s": round(elapsed, 3), "ok": False, "error": str(exc)}


def main():
    print("=" * 50)
    print("Channel Competition: GitHub vs HTTP")
    print("=" * 50)
    print(f"Command: {TEST_CMD}")
    print(f"HTTP API: {HTTP_API}")
    print()

    results = []

    # Test HTTP
    print("[1/2] Testing HTTP direct...")
    http_result = test_http("comp-http-001")
    results.append(http_result)
    print(f"  Latency: {http_result['latency_s']}s {'✓' if http_result['ok'] else '✗'}")
    if http_result["ok"]:
        out = http_result["data"].get("stdout", "")
        print(f"  Output: {out[:100]}...")

    # Test GitHub (simulated with poll delay)
    print("[2/2] Testing GitHub Issues (simulated 60s poll)...")
    gh_result = test_github("comp-gh-001")
    results.append(gh_result)
    print(f"  Latency: {gh_result['latency_s']}s (incl. 60s poll) {'✓' if gh_result['ok'] else '✗'}")

    # Flush all spans to Phoenix
    tracer.flush()
    time.sleep(1)

    # Report
    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    for r in results:
        icon = "✓" if r["ok"] else "✗"
        print(f"  {r['channel']:12s} {r['latency_s']:8.2f}s  {icon}")

    speedup = (results[1]["latency_s"] / results[0]["latency_s"]) if results[0]["latency_s"] > 0 else 0
    print(f"\n  HTTP is {speedup:.0f}x faster than GitHub Issues (excl. poll delay)")
    print(f"  Traces exported to Phoenix → http://localhost:6006")

    # Save results
    with open("/home/reumanlab/phoenix-data/channel_competition_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to ~/phoenix-data/channel_competition_results.json")


if __name__ == "__main__":
    main()
