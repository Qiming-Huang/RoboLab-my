"""Launch a single local Pi0.5 evaluation and retain its status across restarts."""

import contextlib
import fcntl
import json
import os
import re
import subprocess
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dashboard.loaders.catalog import get_task


class LaunchError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)


class RunManager:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.runtime = self.root / ".runtime"
        self.path = self.runtime / "dashboard-jobs.json"
        self.script = self.root / "scripts/local_pi05_eval.sh"
        self._mutex = threading.Lock()
        self._processes: dict[str, subprocess.Popen] = {}

    @contextlib.contextmanager
    def _locked(self):
        self.runtime.mkdir(parents=True, exist_ok=True)
        with self._mutex, (self.runtime / "dashboard-jobs.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            jobs = json.loads(self.path.read_text())
            required = {"id", "task", "run_id", "pid", "status", "log_path"}
            if not isinstance(jobs, list) or any(not isinstance(job, dict) or not required <= job.keys() for job in jobs):
                raise ValueError("invalid saved task runs")
            return jobs
        except (OSError, ValueError) as error:
            raise LaunchError(503, "Could not read saved task runs; check the dashboard logs.") from error

    def _save(self, jobs: list[dict]):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(jobs[-50:], indent=2))
        temporary.replace(self.path)

    def _alive(self, job: dict) -> bool:
        process = self._processes.get(job["id"])
        if process is not None:
            code = process.poll()
            if code is None:
                return True
            job["returncode"] = code
            del self._processes[job["id"]]
            return False
        # A server restart must not allow a second run while its child is alive.
        try:
            arguments = Path(f"/proc/{int(job['pid'])}/cmdline").read_bytes().split(b"\0")
            return job["run_id"].encode() in arguments
        except (OSError, ValueError, KeyError):
            return False

    def _log_tail(self, job: dict) -> str:
        path = self.root / job["log_path"]
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - 16384))
                return stream.read().decode(errors="replace").replace("\r", "\n")
        except OSError:
            return ""

    def _refresh(self, jobs: list[dict]):
        for job in jobs:
            if job["status"] == "running" and not self._alive(job):
                results = self.root / "output" / job["run_id"] / "episode_results.jsonl"
                try:
                    episodes = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
                except (OSError, ValueError):
                    episodes = []
                if episodes and job.get("returncode", 0) == 0:
                    episode = episodes[-1]
                    job.update(status="completed", success=bool(episode.get("success")), score=episode.get("score"))
                else:
                    job.update(status="failed", message="The simulation stopped before a complete result was saved.")
                job["finished_at"] = datetime.now(timezone.utc).isoformat()
            tail = self._log_tail(job)
            matches = re.findall(r"(\d+)%[^\n]*?\|\s*(\d+)/(\d+)\s*\[", tail)
            if matches:
                percent, step, total = matches[-1]
                job["progress"] = {"percent": int(percent), "step": int(step), "total": int(total)}
            if job["status"] == "failed":
                job["log_tail"] = "\n".join(tail.splitlines()[-20:])[-4000:]

    def snapshot(self) -> dict:
        with self._locked():
            jobs = self._load()
            self._refresh(jobs)
            self._save(jobs)
            return {
                "enabled": self.script.is_file() and (self.root / ".venv/bin/python").exists(),
                "jobs": list(reversed(jobs)),
                "active_job": next((j for j in jobs if j["status"] == "running"), None),
            }

    def start(self, name: str) -> dict:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise LaunchError(422, "Choose a task from the benchmark catalog.")
        task = get_task(name)
        if task is None:
            raise LaunchError(404, "Task not found in the benchmark catalog.")
        task_root = (self.root / "robolab/tasks").resolve()
        source = (task_root / task["filename"]).resolve()
        if not source.is_relative_to(task_root) or not source.is_file():
            raise LaunchError(422, "This task's source file is unavailable in the local checkout.")
        folder = source.parent.relative_to(task_root).as_posix()
        with self._locked():
            jobs = self._load()
            self._refresh(jobs)
            if active := next((j for j in jobs if j["status"] == "running"), None):
                raise LaunchError(409, f"A task is already running: {active['task']}.")
            if not self.script.is_file():
                raise LaunchError(503, "The local Pi0.5 launch script is not installed.")
            try:
                port = int(os.environ.get("PI05_PORT", "8010"))
                if not 1 <= port <= 65535:
                    raise ValueError("invalid policy port")
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3) as response:
                    if response.status != 200:
                        raise OSError("policy server is unavailable")
            except (OSError, ValueError, urllib.error.URLError) as error:
                raise LaunchError(503, "Pi0.5 is offline. Start it with bash scripts/local_pi05_server.sh, then retry.") from error
            identifier = uuid.uuid4().hex[:10]
            run_id = f"pi05_{name}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{identifier}"
            log_path = Path(".runtime/logs") / f"{run_id}.log"
            (self.root / log_path).parent.mkdir(parents=True, exist_ok=True)
            (self.root / "output").mkdir(exist_ok=True)
            command = ["bash", str(self.script), "--task", name, "--task-dirs", folder,
                       "--output-folder-name", run_id]
            try:
                with (self.root / log_path).open("wb") as log:
                    process = subprocess.Popen(command, cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, start_new_session=True,
                                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
            except OSError as error:
                raise LaunchError(503, "Could not start the local simulation; check the dashboard logs.") from error
            job = {"id": identifier, "task": name, "run_id": run_id, "pid": process.pid,
                   "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                   "log_path": str(log_path)}
            self._processes[identifier] = process
            jobs.append(job)
            self._save(jobs)
            return job
