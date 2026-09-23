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

def load_reference_solutions():
    ref_file = REPO_ROOT / "reports" / "reference_solutions.json"
    if ref_file.exists():
        try:
            with open(ref_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def load_all_metadata():
    workloads_dir = REPO_ROOT / "workloads"
    meta_dict = {}
    if workloads_dir.exists():
        for case_dir in workloads_dir.glob("case_*"):
            if not case_dir.is_dir():
                continue
            meta_file = case_dir / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r") as f:
                        meta_dict[case_dir.name] = json.load(f)
                except Exception:
                    pass
    return meta_dict

def match_workload(mps_path, model_name):
    path_str = str(mps_path).replace("\\", "/")
    # Match by directory name
    for cid in ["case_a_crude_blending", "case_b_multi_period_planning", "case_c_refinery_scheduling", "case_d_unit_commitment"]:
        if cid in path_str:
            stem = Path(path_str).stem
            if stem == "model":
                stem = "T10" if "case_b" in cid else "Small"
            return cid, stem

    # Match by model name
    m = (model_name or "").upper()
    if "BLENDING" in m or "CRUDE" in m:
        scale = "Small" if "SMALL" in m else ("Medium" if "MED" in m else ("Large" if "LARGE" in m else "Small"))
        return "case_a_crude_blending", scale
    if "PLANNING" in m or "MULTI_PERIOD" in m:
        for t in ["T100", "T50", "T25", "T10"]:
            if t in m:
                return "case_b_multi_period_planning", t
        return "case_b_multi_period_planning", "T10"
    if "REFINERY" in m or "SCHED" in m:
        scale = "Small" if "SMALL" in m else ("Medium" if "MED" in m else ("Large" if "LARGE" in m else "Small"))
        return "case_c_refinery_scheduling", scale
    if "UNIT_COMMIT" in m or "COMMIT" in m:
        scale = "Small" if "SMALL" in m else ("Medium" if "MED" in m else ("Large" if "LARGE" in m else "Small"))
        return "case_d_unit_commitment", scale
    return None, None

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
        elif path == "/api/formulation":
            self._handle_formulation(parsed.query)
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

    def _handle_formulation(self, query_string):
        params = urllib.parse.parse_qs(query_string)
        case_id = params.get("case", [""])[0]
        if not case_id:
            self._set_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": "Missing 'case' parameter"}).encode())
            return

        allowed_cases = ["case_a_crude_blending", "case_b_multi_period_planning", "case_c_refinery_scheduling", "case_d_unit_commitment"]
        if case_id not in allowed_cases:
            self._set_headers(404, "application/json")
            self.wfile.write(json.dumps({"error": "Unknown case ID"}).encode())
            return

        readme_path = REPO_ROOT / "workloads" / case_id / "README.md"
        if not readme_path.exists():
            self._set_headers(404, "application/json")
            self.wfile.write(json.dumps({"error": f"Formulation README not found for {case_id}"}).encode())
            return

        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        self._set_headers(200, "application/json")
        self.wfile.write(json.dumps({"case_id": case_id, "markdown": content}).encode())

    def _handle_samples(self):
        """Discovers existing MPS files in the repository with rich metadata."""
        samples = []
        cases_meta = load_all_metadata()
        ref_solutions = load_reference_solutions()

        # 1. Phase 5 & 6 Industrial Workload Suite
        workloads_dir = REPO_ROOT / "workloads"
        if workloads_dir.exists():
            for case_dir in sorted(workloads_dir.glob("case_*")):
                if not case_dir.is_dir():
                    continue
                cid = case_dir.name
                meta = cases_meta.get(cid, {})
                case_name = meta.get("name", cid.replace("_", " ").title())
                form_class = meta.get("formulation_class", "LP")
                group_label = f"Industrial Suite: {case_name} [{form_class}]"

                for p in sorted(case_dir.glob("*.mps")):
                    rel = p.relative_to(REPO_ROOT)
                    scale = p.stem
                    if scale == "model":
                        continue  # Keep list clean of alias files

                    ref_info = ref_solutions.get(cid, {}).get(scale, {})
                    ref_obj = ref_info.get("objective")

                    samples.append({
                        "name": f"{case_name} — {scale} ({form_class})",
                        "instance": scale,
                        "case_id": cid,
                        "rel_path": str(rel),
                        "abs_path": str(p),
                        "size_bytes": p.stat().st_size,
                        "category": group_label,
                        "formulation_class": form_class,
                        "highs_ref_objective": ref_obj,
                        "provenance_benchmark": meta.get("provenance", {}).get("benchmark", "")
                    })

        # 2. Netlib Benchmark Suite
        netlib_dir = REPO_ROOT / "tests" / "data" / "mps" / "netlib"
        if netlib_dir.exists():
            for p in sorted(netlib_dir.glob("*.mps")):
                rel = p.relative_to(REPO_ROOT)
                samples.append({
                    "name": f"{p.stem} (Netlib LP)",
                    "instance": p.stem,
                    "rel_path": str(rel),
                    "abs_path": str(p),
                    "size_bytes": p.stat().st_size,
                    "category": "Netlib Benchmark Suite (Standard LPs)"
                })

        # 3. General MPS Tests
        general_dir = REPO_ROOT / "tests" / "data" / "mps"
        if general_dir.exists():
            for p in sorted(general_dir.glob("*.mps")):
                if "netlib" in str(p):
                    continue
                rel = p.relative_to(REPO_ROOT)
                samples.append({
                    "name": f"{p.stem} (Test Model)",
                    "instance": p.stem,
                    "rel_path": str(rel),
                    "abs_path": str(p),
                    "size_bytes": p.stat().st_size,
                    "category": "General MPS Unit Tests"
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
        milp_config = "advanced"

        # Check query parameters for optional defaults
        parsed_qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if "milp_config" in parsed_qs and parsed_qs["milp_config"]:
            milp_config = parsed_qs["milp_config"][0].strip().lower()
        if "max_iters" in parsed_qs and parsed_qs["max_iters"]:
            val = parsed_qs["max_iters"][0].strip()
            if val.isdigit():
                max_iters = int(val)

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
                    elif b'name="max_iters"' in part:
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            val_str = part[header_end + 4:].rstrip(b"\r\n").decode("utf-8", errors="ignore").strip()
                            if val_str.isdigit():
                                max_iters = int(val_str)
                    elif b'name="milp_config"' in part:
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            val_str = part[header_end + 4:].rstrip(b"\r\n").decode("utf-8", errors="ignore").strip()
                            if val_str:
                                milp_config = val_str.lower()

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
                if "milp_config" in payload:
                    milp_config = str(payload["milp_config"]).strip().lower()
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

            # Execute real C++/CUDA PipePye binary with Phase 7 MILP configuration
            cmd = [str(RUNNER_BIN), mps_filepath, "--max-iters", str(max_iters), "--milp-config", milp_config]
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

            # Enrich output JSON with Phase 6 HiGHS reference solutions & provenance
            try:
                output_json = json.loads(proc.stdout)
                case_id, instance_key = match_workload(mps_filepath, output_json.get("model", {}).get("name"))

                if case_id and instance_key:
                    ref_data = load_reference_solutions()
                    cases_meta = load_all_metadata()

                    case_ref = ref_data.get(case_id, {}).get(instance_key)
                    case_meta = cases_meta.get(case_id, {})

                    if case_ref:
                        ref_obj = case_ref.get("objective")
                        pipe_obj = output_json.get("executive_summary", {}).get("best_objective")

                        rel_gap = None
                        is_verified = False
                        if pipe_obj is not None and ref_obj is not None and not (isinstance(ref_obj, str) and ref_obj == "Infinity"):
                            try:
                                p_val = float(pipe_obj)
                                r_val = float(ref_obj)
                                rel_gap = abs(p_val - r_val) / max(1.0, abs(r_val))
                                is_verified = (rel_gap < 1e-4)
                            except (ValueError, TypeError):
                                pass

                        output_json["phase6_reference_verification"] = {
                            "has_reference": True,
                            "case_id": case_id,
                            "instance_key": instance_key,
                            "solver": case_ref.get("solver", "HiGHS-1.8.1"),
                            "model_status": case_ref.get("model_status", "Optimal"),
                            "reference_objective": ref_obj,
                            "recomputed_objective": case_ref.get("recomputed_objective"),
                            "objective_discrepancy": case_ref.get("objective_discrepancy"),
                            "highs_simplex_iterations": case_ref.get("performance", {}).get("simplex_iterations", 0),
                            "highs_mip_nodes": case_ref.get("performance", {}).get("mip_nodes", 0),
                            "highs_wallclock_sec": case_ref.get("performance", {}).get("wallclock_time_sec", 0.0),
                            "highs_run_time_sec": case_ref.get("performance", {}).get("highs_run_time_sec", 0.0),
                            "relative_gap_vs_highs": rel_gap,
                            "is_verified_optimal": is_verified,
                            "independent_verification": case_ref.get("independent_verification", {})
                        }
                    else:
                        output_json["phase6_reference_verification"] = {"has_reference": False}

                    if case_meta:
                        matched_inst_meta = {}
                        for inst in case_meta.get("instances", []):
                            if inst.get("scale") == instance_key or inst.get("file_name") == f"{instance_key}.mps":
                                matched_inst_meta = inst
                                break

                        output_json["workload_provenance"] = {
                            "has_provenance": True,
                            "case_id": case_id,
                            "name": case_meta.get("name"),
                            "category": case_meta.get("category"),
                            "formulation_class": case_meta.get("formulation_class"),
                            "mathematical_structure": case_meta.get("mathematical_structure"),
                            "provenance_benchmark": case_meta.get("provenance", {}).get("benchmark"),
                            "references": case_meta.get("provenance", {}).get("references", []),
                            "pre_registered_strategy": case_meta.get("pre_registered_strategy", {}),
                            "instance_scale": instance_key,
                            "ranges": matched_inst_meta.get("ranges", {}),
                            "structural_metrics": matched_inst_meta.get("structural_metrics", {})
                        }
                    else:
                        output_json["workload_provenance"] = {"has_provenance": False}
                else:
                    output_json["phase6_reference_verification"] = {"has_reference": False}
                    output_json["workload_provenance"] = {"has_provenance": False}

                resp_bytes = json.dumps(output_json).encode("utf-8")
                self._set_headers(200, "application/json")
                self.wfile.write(resp_bytes)
                return
            except json.JSONDecodeError:
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
