import json
import subprocess
import sys
import urllib.error
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import dashboard.app as app_module
import dashboard.jobs as jobs_module
from dashboard.jobs import RunManager


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    script = root / "scripts/local_pi05_eval.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n")
    source = root / "robolab/tasks/benchmark/banana_in_bowl_task.py"
    source.parent.mkdir(parents=True)
    source.touch()
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    manager = RunManager(root)
    monkeypatch.setattr(app_module, "RunManager", lambda root: manager)
    process = MagicMock(pid=12345678)
    process.poll.return_value = None
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(jobs_module.subprocess, "Popen", popen)
    health = MagicMock()
    health.return_value.__enter__.return_value.status = 200
    monkeypatch.setattr(jobs_module.urllib.request, "urlopen", health)
    client = TestClient(app_module.create_app())
    return client, manager, process, popen, health


def test_launch_uses_catalog_task_and_argument_array(setup):
    client, manager, process, popen, health = setup
    response = client.post("/api/jobs", json={"task": "BananaInBowlTask"}, headers={"Origin": "http://testserver"})
    assert response.status_code == 202
    job = response.json()
    args, kwargs = popen.call_args
    assert args[0] == ["bash", str(manager.script), "--task", "BananaInBowlTask",
                       "--task-dirs", "benchmark", "--output-folder-name", job["run_id"]]
    assert not kwargs.get("shell", False)
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    assert json.loads(manager.path.read_text())[0]["pid"] == process.pid
    assert client.get("/api/jobs").json()["active_job"]["task"] == "BananaInBowlTask"


@pytest.mark.parametrize("body, status", [
    ({"task": "DoesNotExistTask"}, 404),
    ({"task": "BananaInBowlTask; touch /tmp/unwanted"}, 422),
    ({"task": "../../etc/passwd"}, 422),
    ({"task": "BananaInBowlTask", "command": "arbitrary"}, 422),
])
def test_rejects_unknown_tasks_and_arbitrary_commands(setup, body, status):
    client, _, _, popen, _ = setup
    assert client.post("/api/jobs", json=body).status_code == status
    popen.assert_not_called()


def test_rejects_other_browser_origins(setup):
    client, _, _, popen, _ = setup
    response = client.post("/api/jobs", json={"task": "BananaInBowlTask"}, headers={"Origin": "https://other.example"})
    assert response.status_code == 403
    popen.assert_not_called()


def test_refuses_duplicate_run(setup):
    client, _, _, popen, _ = setup
    assert client.post("/api/jobs", json={"task": "BananaInBowlTask"}).status_code == 202
    assert client.post("/api/jobs", json={"task": "BananaInBowlTask"}).status_code == 409
    assert popen.call_count == 1


def test_reports_offline_policy_without_launching(setup):
    client, _, _, popen, health = setup
    health.side_effect = urllib.error.URLError("connection refused")
    response = client.post("/api/jobs", json={"task": "BananaInBowlTask"})
    assert response.status_code == 503
    assert "Pi0.5 is offline" in response.json()["detail"]
    popen.assert_not_called()


def test_corrupt_registry_does_not_allow_a_new_run(setup):
    client, manager, _, popen, _ = setup
    manager.runtime.mkdir()
    manager.path.write_text('{"unexpected": "state"}')
    assert client.get("/api/jobs").status_code == 503
    assert client.post("/api/jobs", json={"task": "BananaInBowlTask"}).status_code == 503
    popen.assert_not_called()


def test_finished_episode_is_distinct_from_task_success(setup):
    client, manager, process, _, _ = setup
    job = client.post("/api/jobs", json={"task": "BananaInBowlTask"}).json()
    output = manager.root / "output" / job["run_id"]
    output.mkdir()
    (output / "episode_results.jsonl").write_text(json.dumps({"success": False, "score": 0.5}) + "\n")
    process.poll.return_value = 0
    finished = client.get("/api/jobs").json()
    assert finished["active_job"] is None
    assert finished["jobs"][0]["status"] == "completed"
    assert finished["jobs"][0]["success"] is False
    assert finished["jobs"][0]["score"] == 0.5


def test_progress_and_crash_log_are_exposed(setup):
    client, manager, process, _, _ = setup
    job = client.post("/api/jobs", json={"task": "BananaInBowlTask"}).json()
    (manager.root / job["log_path"]).write_text("\r 25%|xx | 10/40 [00:03<00:09]\nRuntimeError: test failure\n")
    assert client.get("/api/jobs").json()["active_job"]["progress"] == {"percent": 25, "step": 10, "total": 40}
    process.poll.return_value = 1
    finished = client.get("/api/jobs").json()["jobs"][0]
    assert finished["status"] == "failed"
    assert "RuntimeError: test failure" in finished["log_tail"]


def test_restart_keeps_a_live_run_exclusive(tmp_path):
    # A real lightweight process verifies that persisted PIDs survive a manager
    # restart without starting a simulator during the test.
    manager = RunManager(tmp_path)
    run_id = "pi05_restart_test"
    process = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()", run_id], stdin=subprocess.PIPE)
    try:
        manager.runtime.mkdir(parents=True)
        manager.path.write_text(json.dumps([{"id": "restart", "task": "BananaInBowlTask", "run_id": run_id,
                                            "pid": process.pid, "status": "running", "log_path": ".runtime/test.log"}]))
        restored = RunManager(tmp_path)
        assert restored.snapshot()["active_job"]["pid"] == process.pid
        process.stdin.close()
        process.wait(timeout=5)
        assert restored.snapshot()["jobs"][0]["status"] == "failed"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
