#!/usr/bin/env python3.11
"""
Lightweight HTTP task server for cluster ↔ reumanlab communication.
Competes with GitHub Issues as a low-latency channel.

Endpoints:
  POST /task      - Submit a shell command, get response
  GET /health     - Health check
  GET /stream     - SSE stream of task output (for real-time feedback)

Usage:
  python3.11 cluster_api.py --port 8888

Deploy:
  sbatch --partition=kbs --time=24:00:00 --mem=4G --cpus-per-task=2 \
    --output=/home/a474r867/scratch/logs/cluster_api_%j.out \
    --wrap="python3.11 /home/a474r867/scratch/nemotron-eco-reasoner/scripts/cluster_api.py --port 8888"
"""

import json
import os
import subprocess
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PORT = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 8888


class TaskHandler(BaseHTTPRequestHandler):
    """Handles POST /task and GET /health."""

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._json(200, {
                "status": "ok",
                "host": os.uname().nodename,
                "pid": os.getpid(),
                "uptime": time.time() - START_TIME,
            })
        elif path == "/stream":
            self._stream_response()
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/task":
            self._json(404, {"error": "not found"})
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}

        cmd = body.get("cmd", "")
        timeout = body.get("timeout", 300)
        task_id = body.get("task_id", str(uuid.uuid4())[:8])

        if not cmd:
            self._json(400, {"error": "missing 'cmd' field"})
            return

        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.expanduser("~/scratch"),
            )
            elapsed = time.time() - start
            self._json(200, {
                "task_id": task_id,
                "exit_code": result.returncode,
                "stdout": result.stdout[-50000:],  # last 50KB
                "stderr": result.stderr[-10000:],
                "elapsed_s": round(elapsed, 3),
                "truncated": len(result.stdout) > 50000,
            })
        except subprocess.TimeoutExpired:
            self._json(504, {
                "task_id": task_id,
                "error": f"timeout after {timeout}s",
                "elapsed_s": round(time.time() - start, 3),
            })
        except Exception as exc:
            self._json(500, {
                "task_id": task_id,
                "error": str(exc),
                "elapsed_s": round(time.time() - start, 3),
            })

    def _stream_response(self):
        """Server-Sent Events stream for real-time task output."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Send a heartbeat every 5s to keep connection alive
        for i in range(12):  # 60 seconds max
            self.wfile.write(f"data: {json.dumps({'ts': time.time(), 'heartbeat': True})}\n\n".encode())
            self.wfile.flush()
            time.sleep(5)

    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        # Minimal logging to stdout for Slurm output capture
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}", flush=True)


START_TIME = time.time()

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), TaskHandler)
    print(f"Cluster API listening on :{PORT} (PID {os.getpid()})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...", flush=True)
        server.shutdown()
