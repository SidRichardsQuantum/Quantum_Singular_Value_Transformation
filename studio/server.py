"""Local, single-user HTTP API and serial scientific worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import mimetypes
import os
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .adapter import execute_request
from .catalogue import catalogue, validate_request
from .records import Store, canonical_json, compare, metrics

STATIC = Path(__file__).parent / "static"


def environment():
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    return {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("qsvt-pennylane", "numpy", "pennylane", "matplotlib")
        },
        "git_commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "scientific_source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted((root / "src/qsvt").glob("*.py")))
        ).hexdigest(),
    }


class Studio:
    def __init__(self, root):
        self.store = Store(root)
        self.store.recover()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qsvt")
        self.submission_lock = threading.Lock()

    def submit(self, raw):
        request = validate_request(raw)
        with self.submission_lock:
            if (
                sum(
                    r["status"] not in {"completed", "failed"}
                    for r in self.store.history()
                )
                >= 8
            ):
                raise ValueError("Queue is full (8 runs); wait for a run to finish.")
            run_id = self.store.create(request, environment())
            self.worker.submit(self.run, run_id)
        return self.store.read(run_id)

    def run(self, run_id):
        started = time.perf_counter()
        try:
            self.store.update(run_id, status="validating")
            request = validate_request(self.store.reuse(run_id))
            self.store.update(run_id, status="executing")
            scientific_start = time.perf_counter()
            report = execute_request(request)
            runtime = time.perf_counter() - scientific_start
            self.store.update(
                run_id, status="saving_artifacts", package_call_seconds=runtime
            )
            directory = self.store.directory(run_id)
            self.store.write(directory / "report.json", report)
            stored = self.store.read(run_id, report=True)
            # A plotting failure must not erase successfully computed science.
            artifacts = ["request.json", "report.json"]
            warnings = []
            try:
                from .plots import render

                (directory / "preview.png").write_bytes(render([stored]))
                artifacts.append("preview.png")
            except Exception as exc:
                warnings.append(f"Preview unavailable: {type(exc).__name__}: {exc}")
            self.store.update(
                run_id,
                status="completed",
                artifacts=artifacts,
                metrics=metrics(stored["report"]),
                warnings=warnings,
                studio_elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as exc:
            self.store.update(
                run_id,
                status="failed",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                studio_elapsed_seconds=time.perf_counter() - started,
            )

    def close(self):
        self.worker.shutdown(wait=True)


def handler(studio):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, data, content_type="application/json"):
            body = canonical_json(data).encode() if isinstance(data, dict) else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob:; "
                "style-src 'self'; script-src 'self'; "
                "object-src 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                self.get()
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self.send(400, {"error": str(exc)})
            except FileNotFoundError:
                self.send(404, {"error": "Run or artifact not found."})

        def get(self):
            url = urlsplit(self.path)
            path = url.path
            if path == "/api/catalogue":
                return self.send(200, catalogue())
            if path == "/api/runs":
                return self.send(200, {"runs": studio.store.history()})
            if path == "/api/compare.png":
                from .plots import render

                ids = parse_qs(url.query).get("id", [])
                return self.send(200, render(compare(studio.store, ids)), "image/png")
            parts = path.strip("/").split("/")
            if parts[:2] == ["api", "runs"] and len(parts) in (3, 4):
                run_id = parts[2]
                if len(parts) == 3:
                    return self.send(200, studio.store.read(run_id, report=True))
                filename = parts[3]
                if filename == "reuse":
                    return self.send(200, studio.store.reuse(run_id))
                if filename not in {"request.json", "report.json", "preview.png"}:
                    raise FileNotFoundError
                file = studio.store.directory(run_id) / filename
                if file.is_symlink():
                    raise ValueError("Symlink artifacts are not supported.")
                return self.send(
                    200,
                    file.read_bytes(),
                    "image/png" if filename.endswith("png") else "application/json",
                )
            filename = "index.html" if path == "/" else path.removeprefix("/")
            if filename not in {"index.html", "app.js", "style.css"}:
                raise FileNotFoundError
            file = STATIC / filename
            self.send(
                200,
                file.read_bytes(),
                mimetypes.guess_type(file.name)[0] or "text/plain",
            )

        def do_POST(self):
            try:
                if (
                    self.headers.get("X-QSVT-Studio") != "1"
                    or self.headers.get("Sec-Fetch-Site") == "cross-site"
                ):
                    return self.send(
                        403, {"error": "Same-origin studio requests required."}
                    )
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected application/json.")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Request must be 1–65536 bytes.")
                raw = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == "/api/runs":
                    return self.send(202, studio.submit(raw))
                if path == "/api/validate":
                    return self.send(200, validate_request(raw))
                if path == "/api/compare":
                    runs = compare(studio.store, raw.get("ids"))
                    return self.send(
                        200,
                        {
                            "runs": [
                                {
                                    "id": r["id"],
                                    "request": r["request"],
                                    "metrics": metrics(r["report"]),
                                    "package_call_seconds": r.get(
                                        "package_call_seconds"
                                    ),
                                }
                                for r in runs
                            ]
                        },
                    )
                parts = path.strip("/").split("/")
                if (
                    len(parts) == 4
                    and parts[:2] == ["api", "runs"]
                    and parts[3] == "favorite"
                ):
                    studio.store.favorite(parts[2], raw.get("favorite"))
                    return self.send(200, studio.store.read(parts[2]))
                self.send(404, {"error": "Unknown endpoint."})
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                self.send(400, {"error": str(exc)})
            except FileNotFoundError:
                self.send(404, {"error": "Run not found."})

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description="QSVT Experiment Studio (local, single user)"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=Path(".qsvt-studio/runs"))
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qsvt-studio-matplotlib")
    # Codespaces/Linux: prevent a second server recovering an active worker's runs.
    import fcntl

    args.data_dir.mkdir(parents=True, exist_ok=True)
    with (args.data_dir / ".server.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another studio server is using this data directory.")
        studio = Studio(args.data_dir)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(studio))
        print(f"QSVT Experiment Studio: http://127.0.0.1:{args.port}", flush=True)
        print(f"Experiment records: {args.data_dir.resolve()}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Waiting for queued scientific work to finish…", flush=True)
        finally:
            server.server_close()
            studio.close()


if __name__ == "__main__":
    main()
