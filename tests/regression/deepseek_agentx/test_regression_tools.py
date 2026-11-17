# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Offline failure-oriented tests; no NPU or service environment is needed."""

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import urllib.request
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from deepseek_agentx import assessment, engine, identity, sources, tasks  # noqa: E402
from deepseek_agentx.common import RegressionError, exclusive_lock, load_profile, read_json, write_json  # noqa: E402
from deepseek_agentx.traffic import observe_traffic  # noqa: E402


@pytest.fixture
def run_dir(tmp_path):
    root = tmp_path / "run"
    (root / "server/ascend/debug/device-12").mkdir(parents=True)
    (root / "agentx/report").mkdir(parents=True)
    (root / "replay").mkdir()
    (root / "environment").mkdir()
    config = {"profile": {"agentx": {"duration_seconds": 1800, "warmup_requests": 1}}}
    write_json(root / "run.json", config)
    (root / "server/server.log").write_text(
        "request warm received: prompt=10 tokens, max_new_tokens=1\n"
        "request warm finished: prompt=10 out=1 reason=FINISHED_LENGTH e2e=1s\n"
        "request formal received: prompt=12 tokens, max_new_tokens=2\n"
        "prefix_cache_hit request_id=formal prefix_cache_hit_tokens=8\n"
        "request formal finished: prompt=12 out=2 reason=FINISHED_LENGTH e2e=1s\n")
    (root / "server/ascend/debug/device-12/device.log").write_text("healthy\n")
    rows, traffic = [], []
    for request, phase, prompt, output in [("warm", "warmup", 10, 1), ("formal", "profiling", 12, 2)]:
        rows.append({"metadata": {"x_request_id": "client-" + request, "benchmark_phase": phase,
                                  "was_cancelled": False, "context_overflow_skip": False},
                     "metrics": {"usage_completion_tokens": {"value": output},
                                 "usage_prompt_tokens": {"value": prompt}}})
        traffic.append({"client_request_id": "client-" + request, "server_request_id": request, "status": 200,
                        "done": True, "finish_reason": "length",
                        "usage": {"prompt_tokens": prompt, "completion_tokens": output}})
    for name, values in (("agentx/report/profile_export.jsonl", rows), ("replay/requests.jsonl", traffic)):
        (root / name).write_text("".join(json.dumps(item) + "\n" for item in values))
    write_json(root / "agentx/report/profile_export_aiperf.json", {
        "schema_version": "1.4", "benchmark_duration": {"avg": 1801}, "request_count": {"avg": 1},
        "was_cancelled": False, "error_summary": [], "request_error_rate": {"avg": 0}})
    write_json(root / "identity-checks.json", {"checks": dict.fromkeys(identity.KEYS, True)})
    write_json(root / "environment/source-after.json", {"verified": True})
    write_json(root / "generated-artifacts.json", {"verified": True})
    write_json(root / "shutdown.json", {"verified": True})
    write_json(root / "agentx-final.json", {"state": "completed", "exit_code": 0, "natural": True})
    write_json(root / "finish-reason.json", {"outcome": "COMPLETED"})
    return root


def change(root, name, **updates):
    value = read_json(root / name)
    value.update(updates)
    write_json(root / name, value)


def test_complete_evidence_passes_stability_only(run_dir):
    result = assessment.assess(run_dir)
    assert result["outcome"] == "PASS_STABILITY"
    assert result["correctness"] == result["performance_comparison"] == "NOT_RUN"
    assert result["actual_prefix_hits"] == 1


@pytest.mark.parametrize("name", ["agentx/report/profile_export_aiperf.json", "agentx/report/profile_export.jsonl",
                                   "identity-checks.json", "environment/source-after.json", "shutdown.json",
                                   "replay/requests.jsonl", "agentx-final.json"])
def test_missing_evidence_cannot_pass(run_dir, name):
    (run_dir / name).unlink()
    assert assessment.assess(run_dir)["outcome"] == "INCOMPLETE"


@pytest.mark.parametrize("payload", ["{broken", "[]", "null", '"not-an-object"'])
def test_invalid_summary_produces_result_instead_of_crashing(run_dir, payload):
    (run_dir / "agentx/report/profile_export_aiperf.json").write_text(payload)
    assert assessment.assess(run_dir)["outcome"] == "INCOMPLETE"


@pytest.mark.parametrize("name,updates", [
    ("agentx/report/profile_export_aiperf.json", {"benchmark_duration": {"avg": 1799}}),
    ("agentx/report/profile_export_aiperf.json", {"schema_version": "99"}),
    ("agentx/report/profile_export_aiperf.json", {"was_cancelled": True}),
    ("agentx/report/profile_export_aiperf.json", {"request_error_rate": {"avg": float("nan")}}),
    ("agentx-final.json", {"natural": False}),
    ("shutdown.json", {"verified": False}),
])
def test_early_cancelled_or_unverified_run_cannot_pass(run_dir, name, updates):
    change(run_dir, name, **updates)
    assert assessment.assess(run_dir)["outcome"] == "INCOMPLETE"


def test_same_request_counts_but_wrong_identity_do_not_pass(run_dir):
    path = run_dir / "replay/requests.jsonl"
    path.write_text(path.read_text().replace("client-formal", "client-someone-else"))
    assert assessment.assess(run_dir)["outcome"] == "INCOMPLETE"


def test_http_200_without_terminal_sse_does_not_pass(run_dir):
    path = run_dir / "replay/requests.jsonl"
    path.write_text(path.read_text().replace('"done": true', '"done": false'))
    assert assessment.assess(run_dir)["outcome"] == "INCOMPLETE"


def test_mte_on_any_card_wins_over_later_completion(run_dir):
    path = run_dir / "server/ascend/debug/device-12/device.log"
    path.write_text("DDR address of the MTE instruction is out of range\n")
    result = assessment.assess(run_dir)
    assert result["outcome"] == "FAIL_DEVICE"
    assert "device-12" in result["first_fault"]["file"]


def test_warning_and_long_prefill_are_not_early_failure(run_dir):
    path = run_dir / "server/server.log"
    path.write_text(path.read_text() + "WARNING long prefill elapsed=31s still running\n")
    assert assessment.assess(run_dir)["outcome"] == "PASS_STABILITY"


def test_fault_split_across_log_writes_is_still_detected(run_dir):
    path = run_dir / "server/server.log"
    scanner = assessment.FaultScanner(run_dir)
    assert scanner.scan() is None
    with path.open("a") as output:
        output.write("DDR address of the MTE instruction is out")
    assert scanner.scan() is None
    with path.open("a") as output:
        output.write(" of range\n")
    assert scanner.scan()["outcome"] == "FAIL_DEVICE"


def test_worker_host_failure_is_classified_separately(run_dir):
    with (run_dir / "server/server.log").open("a") as output:
        output.write("Worker process failed: host scheduling error\n")
    assert assessment.assess(run_dir)["outcome"] == "FAIL_SERVER"


def test_setup_block_keeps_its_classification(run_dir):
    change(run_dir, "finish-reason.json", outcome="BLOCKED_FETCH", detail="network unavailable")
    assert assessment.assess(run_dir)["outcome"] == "BLOCKED_FETCH"


def test_usage_disagreement_is_a_request_failure(run_dir):
    path = run_dir / "replay/requests.jsonl"
    path.write_text(path.read_text().replace('"completion_tokens": 2', '"completion_tokens": 3'))
    assert assessment.assess(run_dir)["outcome"] == "FAIL_REQUESTS"


def test_full_profile_and_long_run_budgets(tmp_path):
    profile = load_profile()
    assert profile["server"]["devices"] == 16
    profile["agentx"]["duration_seconds"] = 7200
    path = tmp_path / "profile.json"
    write_json(path, profile)
    with pytest.raises(ValueError, match="Client budget"):
        load_profile(path)


def test_server_budget_includes_client_queue_wait(tmp_path):
    profile = load_profile()
    profile["budgets"]["server_seconds"] = 10800
    path = tmp_path / "profile.json"
    write_json(path, profile)
    with pytest.raises(ValueError, match="client queue"):
        load_profile(path)


@pytest.fixture
def ci_tree(tmp_path):
    root = tmp_path / "lib"
    for path in (".github/workflows", ".github/actions/setup-ci-job", ".github/actions/pypto-serving-tests"):
        (root / path).mkdir(parents=True)
    (root / ".github/workflows/ci.yml").write_text('''jobs:
  serving-dspark:
    steps:
      - uses: ./.github/actions/setup-ci-job
        with:
          needs-device: 'true'
          runtime-platform: a2a3
          torch-version: torch==2.10.0
      - uses: ./.github/actions/pypto-serving-tests
        with:
          model-name: deepseek-dspark
''')
    (root / ".github/actions/setup-ci-job/action.yml").write_text('''inputs:
  pypto-ref:
    default: ''
  install-torch-npu:
    default: 'false'
# source https://github.com/hw-native-sys/pypto.git
# wheel cp310-cp310-manylinux_2_34_
''')
    (root / ".github/actions/pypto-serving-tests/action.yml").write_text('''runs:
  steps:
    - uses: actions/checkout@v4
      with:
        repository: hw-native-sys/pypto-serving
        ref: main
    - name: Override pypto-serving submodule with pypto-lib under test
''')
    return root


def test_ci_defaults_and_overrides_are_resolved(ci_tree):
    plan = sources.parse_ci(ci_tree, "serving-dspark")
    assert plan["pypto_ref"] == "HEAD"
    assert plan["torch"] == "torch==2.10.0"
    assert plan["serving_ref"] == "main"


def test_dynamic_ci_expression_is_blocked_not_guessed(ci_tree):
    path = ci_tree / ".github/workflows/ci.yml"
    path.write_text(path.read_text().replace("torch==2.10.0", "${{ matrix.torch }}"))
    with pytest.raises(RegressionError, match="dynamic CI input"):
        sources.parse_ci(ci_tree, "serving-dspark")


def test_missing_ci_job_is_not_replaced_by_main(ci_tree):
    with pytest.raises(RegressionError, match="Unsupported lib CI"):
        sources.parse_ci(ci_tree, "unknown-job")


def test_local_checkout_is_frozen_and_original_tree_is_preserved(tmp_path):
    original, frozen = tmp_path / "original", tmp_path / "frozen"
    original.mkdir()
    subprocess.run(["git", "init", "-q", original], check=True)
    (original / "model.py").write_text("VERSION = 1\n")
    subprocess.run(["git", "-C", original, "add", "model.py"], check=True)
    subprocess.run(["git", "-C", original, "-c", "user.name=OfflineTest", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "test snapshot"], check=True)
    sha = sources.clone(str(original), "HEAD", frozen)
    expected = sources.source_record(frozen)
    assert expected["commit"] == sha
    manifest = {"sources": {"pypto": expected}, "lib_override": False}
    sources.snapshot_sources(manifest)
    (frozen / "model.py").write_text("VERSION = 2\n")
    with pytest.raises(RegressionError, match="Source identity changed"):
        sources.snapshot_sources(manifest)
    assert (original / "model.py").read_text() == "VERSION = 1\n"


def test_generated_commands_follow_profile_and_quote_paths(tmp_path):
    profile = load_profile()
    manifest = {"run_dir": str(tmp_path / "a path"), "conda_prefix": "/env path", "driver_root": "/driver path",
                "profile": profile, "port": 8232, "model": {"path": "/model path"},
                "tokenizer": {"path": "/tokenizer path"},
                "sources": {"serving": {"path": "/serving path"}, "agentx": {"path": "/agentx path"}}}
    root = Path(manifest["run_dir"])
    root.mkdir()
    server, client = engine.write_scripts(manifest, 8233)
    for path in (server, client):
        subprocess.run(["bash", "-n", path], check=True)
    text = server.read_text()
    assert text.index('exec >') < text.index('source ')
    assert '--enable-prefix-caching' in text and '--no-enable-prefix-caching' not in text
    assert '"${TASK_DEVICE}"' in text
    assert 'http://127.0.0.1:8233' in client.read_text()
    assert 'ignore-whitelist' not in text


def test_environment_lock_is_shared_across_workspaces(tmp_path):
    path = tmp_path / "one-prefix.lock"
    with exclusive_lock(path):
        with pytest.raises(RegressionError, match="Lock held"):
            with exclusive_lock(path):
                pytest.fail("Second holder acquired the environment lock")


def test_unknown_task_state_is_never_cleanup_success(tmp_path, monkeypatch):
    task = "task_20261117_123456_1234"
    write_json(tmp_path / "server-task.json", {"task_id": task})
    monkeypatch.setattr(tasks, "status", lambda _id: {"state": "unknown", "text": "unavailable", "exit_code": None})
    monkeypatch.setattr(tasks.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "error"))
    assert tasks.stop(tmp_path, "server", 0)["verified"] is False


def test_task_starting_during_cancel_is_killed_by_exact_id(tmp_path, monkeypatch):
    task = "task_20261117_123456_1234"
    write_json(tmp_path / "server-task.json", {"task_id": task})
    states = iter(["pending", "running", "completed"])
    monkeypatch.setattr(tasks, "status", lambda _: {"state": next(states), "text": "", "exit_code": None})
    commands = []

    def execute(args, **kwargs):
        commands.append(args)
        assert "TASKQUEUE_IGNORE_WHITELIST" not in kwargs["env"]
        return subprocess.CompletedProcess(args, 1 if args[1] == "--cancel" else 0, "", "")

    monkeypatch.setenv("TASKQUEUE_IGNORE_WHITELIST", "1")
    monkeypatch.setattr(tasks.subprocess, "run", execute)
    monkeypatch.setattr(tasks.time, "sleep", lambda _: None)
    assert tasks.stop(tmp_path, "server", 10)["verified"] is True
    assert commands == [["task-submit", "--cancel", task], ["task-submit", "--kill", task]]


def test_unknown_finish_value_cannot_override_missing_evidence(run_dir):
    (run_dir / "identity-checks.json").unlink()
    change(run_dir, "finish-reason.json", outcome="PASS_STABILITY")
    assert assessment.assess(run_dir)["outcome"] == "INCOMPLETE"


def test_missing_prompt_usage_on_both_sides_cannot_pass(run_dir):
    path = run_dir / "agentx/report/profile_export.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for item in rows:
        item["metrics"].pop("usage_prompt_tokens")
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    path = run_dir / "replay/requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for item in rows:
        item["usage"].pop("prompt_tokens")
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    assert assessment.assess(run_dir)["outcome"] == "INCOMPLETE"


def test_workflow_parses_and_does_not_enable_schedule_by_default():
    workflow = yaml.safe_load((ROOT / ".github/workflows/deepseek_agentx_regression.yml").read_text())
    full = workflow["jobs"]["full"]
    assert "DEEPSEEK_AGENTX_ENABLE_SCHEDULE" in full["if"]
    cleanup = next(step for step in full["steps"] if step.get("name") == "Stop only this run's tasks")
    assert cleanup["if"] == "always()"


def test_cleanup_fallback_writes_incomplete_report_after_controller_loss(run_dir, monkeypatch):
    spec = importlib.util.spec_from_file_location("regression_cli", ROOT / "scripts/run_deepseek_agentx_regression.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    (run_dir / "finish-reason.json").unlink()
    monkeypatch.setattr(sys, "argv", ["regression", "cleanup", "--run-dir", str(run_dir)])
    monkeypatch.setattr(engine, "cleanup", lambda *_: {"verified": True})
    assert cli.main() == 0
    assert read_json(run_dir / "result.json")["outcome"] == "INCOMPLETE"
    assert (run_dir / "result.md").is_file()


def test_every_new_python_file_has_project_header():
    spec = importlib.util.spec_from_file_location("header_guard", ROOT / "tests/lint/check_headers.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    paths = [ROOT / "scripts/run_deepseek_agentx_regression.py", * (ROOT / "scripts/deepseek_agentx").rglob("*.py"),
             Path(__file__)]
    for path in paths:
        assert checker.check_header(path.resolve())[0], str(path)


@pytest.mark.parametrize("mutation,key", [
    ("native", "runtime_native"), ("compiler", "compiler_native"), ("import", "actual_imports"),
    ("isa", "pto_isa"), ("dirty", "source_identity"), ("devices", "allocated_devices"),
])
def test_wrong_worker_identity_is_rejected(tmp_path, mutation, key):
    sources_ = {name: {"path": str(tmp_path / name), "commit": name + "-sha"}
                for name in ("serving", "pypto", "runtime", "lib")}
    manifest = {"sources": sources_, "lib_override": False, "conda_prefix": str(tmp_path / "conda"),
                "pto_isa": "isa-sha", "ptoas": {"version": "0.72"}, "ci": {"torch": "torch==2.10.0"}}
    observed = {"sys_prefix": manifest["conda_prefix"],
        "repositories": {name: {"commit": item["commit"], "status": ""} for name, item in sources_.items()},
        "imports": {module: {"path": str(Path(sources_[repo]["path"]) / "__init__.py"), "sha256": module}
                    for module, repo in (("pypto", "pypto"), ("simpler", "runtime"), ("pypto_serving", "serving"),
                                         ("pypto.pypto_core", "pypto"), ("_task_interface", "runtime"))},
        "environment": {"PYPTO_LIB_ROOT": sources_["lib"]["path"], "TASK_DEVICE": ",".join(map(str, range(16)))},
        "isa": {"commit": "isa-sha", "status": ""},
        "isa_build": {"schema_version": 3, "required_commit_from_pin": "isa-sha", "actual_checkout_commit": "isa-sha",
                      "runtime_artifacts": {"a2a3/host": {"required_commit_from_pin": "isa-sha",
                                                         "actual_checkout_commit": "isa-sha"}}},
        "runtime_libraries": {"a2a3.so": "hash"}, "ptoas": {"sha256": "hash", "version": "ptoas 0.72"},
        "torch": "2.10.0+cpu"}
    observed["imports"]["_task_interface"]["build_commit"] = sources_["runtime"]["commit"]
    built = copy.deepcopy(observed)
    assert all(identity.checks(manifest, observed, built, 1).values())
    if mutation == "native":
        observed["imports"]["_task_interface"]["build_commit"] = "stale-runtime"
    elif mutation == "compiler":
        observed["imports"]["pypto.pypto_core"]["sha256"] = "stale-binary"
    elif mutation == "import":
        observed["imports"]["pypto"]["path"] = "/other/pypto/__init__.py"
    elif mutation == "isa":
        observed["isa_build"]["actual_checkout_commit"] = "other-isa"
    elif mutation == "dirty":
        observed["repositories"]["lib"]["status"] = " M kernel.py"
    else:
        observed["environment"]["TASK_DEVICE"] = "0,1"
    assert identity.checks(manifest, observed, built, 1)[key] is False


def test_streaming_observer_forwards_request_and_joins_sse(tmp_path):
    body = b'{"model":"test","messages":[],"stream":true}'
    payload = (b'data: {"id":"server-42","choices":[{"delta":{"content":"ok"}}]}\n\n'
               b'data: {"id":"server-42","choices":[{"finish_reason":"length"}],'
               b'"usage":{"prompt_tokens":10,"completion_tokens":1}}\n\n'
               b'data: [DONE]\n\n')
    received = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"test"}]}')

        def do_POST(self):
            received.append((self.headers.get("X-Request-ID"), self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(payload[:31])
            self.wfile.flush()
            self.wfile.write(payload[31:])

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        with observe_traffic(tmp_path, upstream.server_port, 5) as port:
            request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                             headers={"X-Request-ID": "client-42", "Content-Type": "application/json"})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                assert json.load(response)["data"][0]["id"] == "test"
            with opener.open(request, timeout=5) as response:
                assert response.read() == payload
        observed = json.loads((tmp_path / "replay/requests.jsonl").read_text())
        assert received == [("client-42", body)]
        assert observed["server_request_id"] == "server-42" and observed["done"] is True
        assert observed["usage"]["completion_tokens"] == 1
        assert (tmp_path / observed["body_file"]).read_bytes() == body
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)


def test_generated_server_device_validation_executes(tmp_path):
    manifest = {"run_dir": str(tmp_path), "conda_prefix": str(Path(sys.executable).parent.parent),
                "driver_root": str(ROOT), "profile": load_profile(), "port": 8232,
                "model": {"path": "/unused"}, "sources": {"serving": {"path": str(tmp_path)}}}
    server, _ = engine.write_scripts(manifest)
    check = server.read_text().split("<<'CHECK'\n", 1)[1].split("\nCHECK", 1)[0]
    valid = subprocess.run([sys.executable, "-", ",".join(map(str, range(16)))], input=check, text=True,
                           capture_output=True)
    invalid = subprocess.run([sys.executable, "-", "0,1"], input=check, text=True, capture_output=True)
    assert valid.returncode == 0
    assert invalid.returncode != 0


@pytest.mark.parametrize("scenario,expected", [("success", "PASS_STABILITY"), ("device", "FAIL_DEVICE"),
                                             ("queue", "BLOCKED_QUEUE"), ("signal", "INCOMPLETE")])
def test_orchestrator_finalizes_own_run_and_holds_lock_through_cleanup(run_dir, tmp_path, monkeypatch,
                                                                    scenario, expected):
    from deepseek_agentx import provision
    work, target = tmp_path / "work", tmp_path / "work/artifacts/dspark-regression/test"
    work.mkdir()
    prefix = tmp_path / "conda"
    cann = tmp_path / "cann"
    cann.mkdir()
    (cann / "set_env.sh").touch()
    args = SimpleNamespace(work_root=work, run_dir=target, conda_env="fake", model=tmp_path / "model",
                           tokenizer=None, cann_root=cann, port=8232, index_url="https://example.invalid/simple")
    monkeypatch.setattr(provision, "conda_environment", lambda _: (prefix, prefix / "conda.sh", "conda"))
    monkeypatch.setattr(provision, "environment_busy", lambda _: [])
    monkeypatch.setattr(engine.shutil, "which", lambda _: "/mock/bin/tool")
    monkeypatch.setattr(engine, "ensure_port_free", lambda _: None)
    monkeypatch.setattr(provision, "model_identity", lambda _: {"path": "/model"})
    monkeypatch.setattr(provision, "tokenizer_identity", lambda _: {"path": "/model"})
    source = {"path": str(tmp_path / "checkout"), "commit": "fake-sha", "date": "2026-11-17"}
    monkeypatch.setattr(sources, "resolve", lambda *_: {"sources": {"serving": source, "pypto": source, "agentx": source},
                                                     "lib_override": False})
    monkeypatch.setattr(sources, "snapshot_sources", lambda _: {})
    monkeypatch.setattr(provision, "activation_environment", lambda _: {})

    def prepare(_manifest):
        engine.shutil.copytree(run_dir, target, dirs_exist_ok=True)
        write_json(target / "environment/built.json", {"imports": {}, "runtime_libraries": {}})

    monkeypatch.setattr(provision, "prepare", prepare)
    monkeypatch.setattr(engine, "run_command", lambda *_a, **_k: write_json(
        target / "environment/after.json", {"imports": {}, "runtime_libraries": {}}))

    def submit(root, role, command, _devices, _budgets):
        if scenario == "queue":
            raise RegressionError("BLOCKED_QUEUE", "fake queue full")
        task = "task_20261117_123456_" + ("1" if role == "server" else "2")
        write_json(root / f"{role}-task.json", {"task_id": task, "command": command})
        return task

    monkeypatch.setattr(tasks, "submit", submit)

    def wait(*_args):
        if scenario == "device":
            (target / "server/ascend/debug/device-12/device.log").write_text("error 507018\n")
            raise RegressionError("FAIL_DEVICE", "fake MTE")
        if scenario == "signal":
            signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(engine, "wait_for", wait)
    stopped = []
    original_cleanup = engine.cleanup

    def cleanup(root, port, budget):
        with pytest.raises(RegressionError, match="Lock held"):
            with exclusive_lock(work / ".deepseek-agentx.lock"):
                pytest.fail("Run lock released before cleanup")
        return original_cleanup(root, port, budget)

    monkeypatch.setattr(engine, "cleanup", cleanup)
    monkeypatch.setattr(tasks, "recover_submission", lambda *_: None)
    monkeypatch.setattr(tasks, "stop", lambda root, role, budget: stopped.append(role) or {"verified": True})
    monkeypatch.setattr(engine, "own_processes", lambda _: ([], []))
    monkeypatch.setattr(engine, "port_listening", lambda _: False)
    exit_code = engine.run(args, load_profile())
    assert exit_code == (0 if scenario == "success" else 1)
    assert read_json(target / "result.json")["outcome"] == expected
    assert read_json(target / "shutdown.json")["verified"] is True
    assert stopped == ["agentx", "server"]
    assert (target / "result.md").is_file() and (target.parent / "summary.md").is_file()
    with exclusive_lock(work / ".deepseek-agentx.lock"):
        pass
