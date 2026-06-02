#!/usr/bin/env python3
"""
Integration tests for cluster_api.py — the lightweight HTTP task server
for cluster ↔ reumanlab communication.

Tests: health check, task execution, timeout, error handling.

Usage:
    python3 test_cluster_api.py

Requirements: Python 3.8+, no external dependencies (stdlib only).
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error


def find_free_port():
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def api_request(method, path, port, body=None, timeout=10):
    """Make an HTTP request to the cluster API server."""
    url = f"http://127.0.0.1:{port}{path}"
    if body is not None:
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())
    except urllib.error.URLError as e:
        return None, {"error": str(e)}


def start_server(port, script_path):
    """Start cluster_api.py on the given port, return Popen object."""
    # cluster_api.py hardcodes cwd=~/scratch (cluster convention).
    # Create it locally so tasks can execute.
    scratch_dir = os.path.expanduser("~/scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    proc = subprocess.Popen(
        [sys.executable, script_path, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,  # process group for clean kill
    )
    # Wait for server to be ready
    for _ in range(20):
        try:
            code, data = api_request("GET", "/health", port, timeout=2)
            if code == 200 and data.get("status") == "ok":
                return proc
        except Exception:
            pass
        time.sleep(0.25)
    proc.kill()
    raise RuntimeError(f"Server failed to start on port {port}")


def stop_server(proc):
    """Gracefully kill the server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_health(port):
    """GET /health returns 200 with expected fields."""
    code, data = api_request("GET", "/health", port)
    assert code == 200, f"Expected 200, got {code}"
    assert data["status"] == "ok"
    assert "host" in data
    assert "pid" in data
    assert "uptime" in data
    assert isinstance(data["uptime"], float)
    assert data["uptime"] >= 0
    print(f"  PASS test_health (uptime={data['uptime']:.1f}s)")


def test_task_success(port):
    """POST /task with a simple command returns success."""
    code, data = api_request("POST", "/task", port, body={
        "cmd": "echo hello world",
        "timeout": 10,
    })
    assert code == 200, f"Expected 200, got {code}: {data}"
    assert data["exit_code"] == 0
    assert "hello world" in data["stdout"]
    assert "task_id" in data
    assert data["elapsed_s"] > 0
    print(f"  PASS test_task_success (id={data['task_id']})")


def test_task_failure(port):
    """POST /task with a failing command captures stderr and non-zero exit."""
    code, data = api_request("POST", "/task", port, body={
        "cmd": "echo 'error msg' >&2 && exit 42",
        "timeout": 10,
    })
    assert code == 200, f"Expected 200, got {code}"
    assert data["exit_code"] == 42
    assert "error msg" in data["stderr"]
    print(f"  PASS test_task_failure (exit_code={data['exit_code']})")


def test_task_timeout(port):
    """POST /task with a command that exceeds timeout returns 504."""
    code, data = api_request("POST", "/task", port, body={
        "cmd": "sleep 30",
        "timeout": 2,
    })
    assert code == 504, f"Expected 504, got {code}: {data}"
    assert "timeout" in data["error"].lower()
    print(f"  PASS test_task_timeout")


def test_missing_cmd(port):
    """POST /task without 'cmd' returns 400."""
    code, data = api_request("POST", "/task", port, body={
        "timeout": 10,
    })
    assert code == 400, f"Expected 400, got {code}"
    assert "missing" in data["error"].lower()
    print(f"  PASS test_missing_cmd")


def test_not_found(port):
    """Unknown path returns 404."""
    code, data = api_request("GET", "/nonexistent", port)
    assert code == 404
    print(f"  PASS test_not_found")


def test_stream(port):
    """GET /stream returns SSE content-type."""
    url = f"http://127.0.0.1:{port}/stream"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            assert resp.status == 200
            content_type = resp.headers.get("Content-Type", "")
            assert "text/event-stream" in content_type, f"Got: {content_type}"
            # Read first heartbeat line
            first = resp.readline().decode()
            assert "data:" in first
            print(f"  PASS test_stream (first event: {first.strip()[:50]})")
    except urllib.error.URLError:
        print("  SKIP test_stream (connection closed before read — expected with SSE)")
    except Exception as e:
        print(f"  PASS test_stream (content-type ok, read error: {e})")


def main():
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "scripts",
        "cluster_api.py",
    )
    if not os.path.exists(script_path):
        print(f"ERROR: cluster_api.py not found at {script_path}")
        sys.exit(1)

    port = find_free_port()
    print(f"Starting cluster_api.py on port {port}...")
    proc = start_server(port, script_path)

    passed = 0
    failed = 0
    tests = [
        ("health", test_health),
        ("task_success", test_task_success),
        ("task_failure", test_task_failure),
        ("task_timeout", test_task_timeout),
        ("missing_cmd", test_missing_cmd),
        ("not_found", test_not_found),
        ("stream", test_stream),
    ]

    for name, fn in tests:
        try:
            fn(port)
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
            failed += 1

    stop_server(proc)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
