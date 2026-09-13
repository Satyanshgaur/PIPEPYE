#!/usr/bin/env python3
"""
PipePye Local Demonstration Server
A minimal zero-dependency local HTTP server bridging browser requests to the
compiled PipePye C++/CUDA optimization pipeline runner (pipepye_dashboard_runner).

Usage:
    python3 dashboard/server.py [--port 8080] [--build-dir build]
"""

import os
import sys
import json
import subprocess
import tempfile
import argparse
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = REPO_ROOT / "dashboard"
BUILD_DIR = REPO_ROOT / "build"
RUNNER_BIN = BUILD_DIR / "bin" / "pipepye_dashboard_runner"

class PipePyeRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(DASHBOARD_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/dashboard.css":
            self._serve_file(DASHBOARD_DIR / "dashboard.css", "text/css; charset=utf-8")
        elif path == "/dashboard.js":
            self._serve_file(DASHBOARD_DIR / "dashboard.js", "application/javascript; charset=utf-8")
        elif path == "/api/samples":
            self._handle_samples()
        elif path == "/api/status":
            self._handle_status()
        else:
            self._set_headers(404, "text/plain")
            self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/solve":
            self._handle_solve()
        else:
            self._set_headers(404, "application/json")
            self.wfile.write(json.dumps({"error": "Unknown endpoint"}).encode())

    def _serve_file(self, filepath: Path, content_type: str):
        if not filepath.exists():
            self._set_headers(404, "text/plain")
            self.wfile.write(f"File not found: {filepath.name}".encode())
            return
        with open(filepath, "rb") as f:
            data = f.read()
        self._set_headers(200, content_type)
        self.wfile.write(data)

    def _handle_status(self):
        binary_exists = RUNNER_BIN.exists()
        self._set_headers(200, "application/json")
        self.wfile.write(json.dumps({
            "status": "ready" if binary_exists else "binary_missing",
            "runner_path": str(RUNNER_BIN),
            "repo_root": str(REPO_ROOT),
        }).encode())

    def _handle_samples(self):
        """Discovers existing MPS files in the repository for quick testing."""
        samples = []
        search_dirs = [
            ("Workloads (Phase 5 Industrial)", REPO_ROOT / "workloads"),
            ("Netlib Test Suite", REPO_ROOT / "tests" / "data" / "mps" / "netlib"),
            ("General MPS Tests", REPO_ROOT / "tests" / "data" / "mps")
        ]

        for category, sdir in search_dirs:
            if not sdir.exists():
                continue
            for p in sorted(sdir.glob("**/*.mps")):
                rel = p.relative_to(REPO_ROOT)
                samples.append({
                    "name": p.stem,
                    "rel_path": str(rel),
                    "abs_path": str(p),
                    "size_bytes": p.stat().st_size,
                    "category": category
                })

        self._set_headers(200, "application/json")
        self.wfile.write(json.dumps({"samples": samples}).encode())

    def _handle_solve(self):
        if not RUNNER_BIN.exists():
            self._set_headers(500, "application/json")
            self.wfile.write(json.dumps({
                "error": f"Binary not found: {RUNNER_BIN}. Please build pipepye_dashboard_runner first."
            }).encode())
            return

        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        tmp_mps_file = None
        mps_filepath = None
        max_iters = 3000

        try:
            if "multipart/form-data" in content_type:
                # Parse multipart file upload
                boundary = content_type.split("boundary=")[-1].encode()
                body = self.rfile.read(content_length)

                # Find file content between boundaries
                parts = body.split(b"--" + boundary)
                file_bytes = None
                filename = "uploaded.mps"

                for part in parts:
                    if b"Content-Disposition" in part and b"filename=" in part:
                        # Extract header vs payload
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            file_bytes = part[header_end + 4:].rstrip(b"\r\n")
                            break

                if not file_bytes:
                    self._set_headers(400, "application/json")
                    self.wfile.write(json.dumps({"error": "No file content in multipart body"}).encode())
                    return

                with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_mps_file = tmp.name
                    mps_filepath = tmp.name

            elif "application/json" in content_type:
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode("utf-8"))
                if "filepath" in payload:
                    mps_filepath = payload["filepath"]
                elif "content" in payload:
                    with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as tmp:
                        tmp.write(payload["content"].encode("utf-8"))
                        tmp_mps_file = tmp.name
                        mps_filepath = tmp.name
                if "max_iters" in payload:
                    max_iters = int(payload["max_iters"])
            else:
                # Raw MPS text or binary
                body = self.rfile.read(content_length)
                with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as tmp:
                    tmp.write(body)
                    tmp_mps_file = tmp.name
                    mps_filepath = tmp.name

            if not mps_filepath or not os.path.exists(mps_filepath):
                self._set_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": "Invalid file or filepath"}).encode())
                return

            # Execute real C++/CUDA PipePye binary
            cmd = [str(RUNNER_BIN), mps_filepath, "--max-iters", str(max_iters)]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                timeout=120
            )

            if proc.returncode != 0 and not proc.stdout.strip().startswith("{"):
                self._set_headers(500, "application/json")
                self.wfile.write(json.dumps({
                    "error": f"Execution failed (code {proc.returncode})",
                    "stderr": proc.stderr,
                    "stdout": proc.stdout
                }).encode())
                return

            # Output is pure JSON
            self._set_headers(200, "application/json")
            self.wfile.write(proc.stdout.encode("utf-8"))

        except subprocess.TimeoutExpired:
            self._set_headers(504, "application/json")
            self.wfile.write(json.dumps({"error": "Solver execution timed out (120s limit)"}).encode())
        except Exception as e:
            self._set_headers(500, "application/json")
            self.wfile.write(json.dumps({"error": str(e)}).encode())
        finally:
            if tmp_mps_file and os.path.exists(tmp_mps_file):
                try:
                    os.remove(tmp_mps_file)
                except OSError:
                    pass

def main():
    parser = argparse.ArgumentParser(description="PipePye Demonstration Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()

    ThreadingHTTPServer.allow_reuse_address = True
    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, PipePyeRequestHandler)
    httpd.daemon_threads = True
    print("=" * 70)
    print(f"  PipePye Demonstration Web Server Running")
    print(f"  Local URL:    http://localhost:{args.port}  or  http://127.0.0.1:{args.port}")
    print(f"  Network URL:  http://192.168.1.12:{args.port}")
    print(f"  Runner binary: {RUNNER_BIN}")
    print("=" * 70)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    main()
