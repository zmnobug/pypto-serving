# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One-run orchestration, evidence capture and task-scoped cleanup."""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import traceback
import urllib.request

from . import assessment, identity, provision, sources, tasks
from .common import ROOT, RegressionError, event, exclusive_lock, now, read_json, run_command, write_json
from .traffic import observe_traffic


def port_listening(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def ensure_port_free(port):
    with socket.socket() as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as exc:
            raise RegressionError("BLOCKED_SETUP", f"Port {port} is occupied") from exc


def healthy(port, model):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            health = json.load(response)
        with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=3) as response:
            models = json.load(response)
        return health.get("status") == "ok" and any(x.get("id") == model for x in models["data"])
    except (OSError, ValueError, KeyError):
        return False


def server_command(manifest):
    s = manifest["profile"]["server"]
    args = [str(Path(manifest["conda_prefix"]) / "bin/python"), "-m", "pypto_serving.cli",
            "--model", manifest["model"]["path"], "--served-model-name", s["model_name"],
            "--backend", "npu", "--platform", "a2a3", "--devices", "${TASK_DEVICE}"]
    for option, key in (("dp", "dp"), ("ep", "ep"), ("tp", "tp"), ("block-size", "block_size"),
                        ("max-model-len", "max_model_len"), ("max-num-seqs", "max_num_seqs"),
                        ("max-num-batched-tokens", "max_num_batched_tokens"),
                        ("long-prefill-token-threshold", "long_prefill_token_threshold"),
                        ("npu-memory-utilization", "memory_utilization")):
        args += ["--" + option, str(s[key])]
    args += ["--enable-prefix-caching", "--speculative-config",
             json.dumps({"method": "dspark", "num_speculative_tokens": s["speculative_tokens"]}),
             "--ring-heap", s["ring_heap"], "--host", "127.0.0.1", "--port", str(manifest["port"]),
             "--show-startup-logs"]
    return args


def client_command(manifest, port):
    root = Path(manifest["run_dir"])
    a, s = manifest["profile"]["agentx"], manifest["profile"]["server"]
    return [str(root / ".venv-agentx/bin/aiperf"), "profile", "--scenario", a["scenario"],
        "--url", f"http://127.0.0.1:{port}", "--model", s["model_name"],
        "--tokenizer", manifest["tokenizer"]["path"], "--endpoint-type", "chat", "--public-dataset", a["dataset"],
        "--max-context-length", str(s["max_model_len"]), "--concurrency", str(a["concurrency"]),
        "--use-server-token-count", "--use-legacy-max-tokens", "--streaming", "--extra-inputs", "ignore_eos:true",
        "--cache-bust", a["cache_bust"], "--benchmark-duration", str(a["duration_seconds"]),
        "--warmup-request-count", str(a["warmup_requests"]),
        "--random-seed", str(a["seed"]), "--artifact-dir", str(root / "agentx/report"), "--ui", "simple"]


def shell_command(args):
    return " ".join('"${TASK_DEVICE}"' if value == "${TASK_DEVICE}" else shlex.quote(str(value)) for value in args)


def write_scripts(manifest, observer_port=None):
    root = Path(manifest["run_dir"])
    server = root / "run-server.sh"
    # Redirection precedes activation: even activation/import failures are retained.
    text = f'''#!/usr/bin/env bash
set -eo pipefail
RUN_DIR={shlex.quote(str(root))}
mkdir -p "$RUN_DIR/server/ascend" "$RUN_DIR/jit"
exec > >(tee "$RUN_DIR/server/server.log") 2>&1
trap 'rc=$?; printf "%s\\n" "$rc" > "$RUN_DIR/server/worker.exit"' EXIT
source {shlex.quote(str(root / 'activate-run.sh'))}
: "${{TASK_DEVICE:?missing task allocation}}"
python - "$TASK_DEVICE" <<'CHECK'
import sys
items = [int(x) for x in sys.argv[1].split(',')]
assert len(items) == len(set(items)) == 16 and min(items) >= 0, items
CHECK
export DSPARK_REGRESSION_MANIFEST={shlex.quote(str(root / 'manifest.json'))}
export PYTHONPATH={shlex.quote(str(Path(manifest['driver_root']) / 'scripts/deepseek_agentx/probe'))}:"$PYTHONPATH"
export PYPTO_WORKER_INIT_TIMEOUT={manifest['profile']['budgets']['startup_seconds']}
cd {shlex.quote(manifest['sources']['serving']['path'])}
{shell_command(server_command(manifest))}
'''
    server.write_text(text)
    client = root / "run-agentx.sh"
    if observer_port is not None:
        text = f'''#!/usr/bin/env bash
set -eo pipefail
RUN_DIR={shlex.quote(str(root))}
exec > >(tee "$RUN_DIR/agentx/console.log") 2>&1
trap 'rc=$?; printf "%s\\n" "$rc" > "$RUN_DIR/agentx/worker.exit"' EXIT
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV DSPARK_REGRESSION_MANIFEST
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export NO_PROXY="127.0.0.1,localhost,${{NO_PROXY:-}}" no_proxy="127.0.0.1,localhost,${{no_proxy:-}}"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=2100
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1 HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
cd {shlex.quote(manifest['sources']['agentx']['path'])}
{shlex.quote(str(root / '.venv-agentx/bin/python'))} -c 'import sys,aiperf,json; print(json.dumps({{"python":sys.executable,"aiperf":aiperf.__file__}}))' > "$RUN_DIR/agentx/client-identity.json"
{shell_command(client_command(manifest, observer_port))}
'''
        client.write_text(text)
    for path in (server, client):
        if path.is_file():
            subprocess.run(["bash", "-n", path], check=True, timeout=10)
    return server, client


def process_is_terminal(path):
    try:
        return bool(re.search(r"^State:\s+Z\b", (path / "status").read_text(), re.MULTILINE))
    except (FileNotFoundError, ProcessLookupError):
        return True


def own_processes(root):
    marker = ("DSPARK_REGRESSION_MANIFEST=" + str(root / "manifest.json")).encode()
    found, uncertain = [], []
    known = {str(read_json(path)["pid"]) for path in (root / "environment/workers").glob("worker-*.json")}
    for path in Path("/proc").glob("[0-9]*"):
        if int(path.name) == os.getpid():
            continue
        try:
            # A terminated child can remain as a zombie until its reaper runs;
            # its environment may already be unreadable, but it cannot use an NPU.
            if path.name in known and process_is_terminal(path):
                continue
            if marker in (path / "environ").read_bytes().split(b"\0"):
                found.append(int(path.name))
        except PermissionError:
            if path.name in known:
                uncertain.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found, uncertain


def cleanup(root, port, budget):
    records = {}
    for role in ("agentx", "server"):
        try:
            tasks.recover_submission(root, role)
            records[role] = tasks.stop(root, role, budget)
        except Exception as exc:
            records[role] = {"verified": False, "error": f"{type(exc).__name__}: {exc}"}
    owns_server = (root / "server-task.json").is_file()
    deadline = time.monotonic() + budget
    while True:
        try:
            remaining, uncertain = own_processes(root)
        except (OSError, ValueError, KeyError) as exc:
            remaining, uncertain = [], [str(exc)]
        listening = port_listening(port) if port and owns_server else False
        if not (remaining or uncertain or listening) or time.monotonic() >= deadline:
            break
        time.sleep(1)
    verified = all(value["verified"] for value in records.values()) and not (remaining or uncertain or listening)
    result = {"schema_version": 1, "at": now(), "tasks": records, "verified": verified,
              "port_listening": listening, "remaining_run_processes": remaining, "unreadable_worker_pids": uncertain}
    write_json(root / "shutdown.json", result)
    return result


def wait_for(root, task, manifest, scanner, role, server_task=None):
    budgets = manifest["profile"]["budgets"]
    queued_at, running_at, heartbeat, identity_since = time.monotonic(), None, 0, None
    while True:
        fault = scanner.scan()
        if fault:
            write_json(root / "first-fault.json", {"detected_at": now(), **fault})
            event(root, "fault_detected", **fault)
            time.sleep(budgets["fault_flush_seconds"])
            raise RegressionError(fault["outcome"], fault["text"])
        current = tasks.status(task)
        state = current["state"]
        if state == "running" and running_at is None:
            running_at = time.monotonic()
            event(root, f"{role}_running", task_id=task)
        if state == "completed":
            if role == "server":
                raise RegressionError("FAIL_SERVER", f"Server exited before readiness: {current}")
            current["natural"] = True
            write_json(root / "agentx-final.json", current)
            if current["exit_code"] != 0:
                raise RegressionError("FAIL_REQUESTS", f"AgentX exited {current['exit_code']}")
            return
        if state == "unknown":
            raise RegressionError("INCOMPLETE", f"Unknown queue state for {task}: {current['text']}")
        if running_at is None and time.monotonic() - queued_at > budgets["queue_seconds"]:
            raise RegressionError("BLOCKED_QUEUE", f"{role} queue budget exceeded")
        limit = budgets["startup_seconds"] if role == "server" else budgets["client_seconds"]
        if running_at is not None and time.monotonic() - running_at > limit:
            raise RegressionError("INCOMPLETE", f"{role} execution budget exceeded")
        if role == "server" and state == "running" and healthy(manifest["port"], manifest["profile"]["server"]["model_name"]):
            if identity_since is None:
                identity_since = time.monotonic()
            checks = identity.validate_workers(manifest, root)
            if all(checks.values()):
                event(root, "server_ready", identity_checks=checks)
                return
            if time.monotonic() - identity_since > 30:
                raise RegressionError("BLOCKED_SETUP", f"Worker identity checks failed: {checks}")
        if server_task and tasks.status(server_task)["state"] != "running":
            raise RegressionError("FAIL_SERVER", "Server left running state during AgentX load")
        if time.monotonic() >= heartbeat:
            event(root, f"waiting_{role}", task_id=task, state=state,
                  stage="execution" if running_at is not None else "queue")
            heartbeat = time.monotonic() + 30
        time.sleep(budgets["poll_seconds"])


def render(root, manifest, result):
    lines = ["# DeepSeek + AgentX regression", "", f"Outcome: **{result['outcome']}**", "",
             f"Run: `{root.name}`", f"Finished: {result['at']}", "",
             "| Component | Commit | Date |", "| --- | --- | --- |"]
    for name, item in manifest.get("sources", {}).items():
        lines.append(f"| {name} | `{item['commit']}` | {item['date']} |")
    lines += ["", f"Server received/finished: {result.get('server_received')}/{result.get('server_finished')}",
              f"Actual prefix hits: {result.get('actual_prefix_hits')}",
              f"Formal seconds: {result.get('profiling_seconds')}",
              "", "Correctness and performance comparison: NOT_RUN. This run assesses stability only.", "",
              "## Findings", ""]
    details = [result.get("finish_reason", {}).get("detail", "")]
    details += result.get("request_errors", []) + result.get("incomplete_reasons", [])
    lines += ["- " + str(x).replace("\n", " ") for x in details if x]
    lines += ["", "## Evidence", "", "- [Frozen manifest](manifest.json)", "- [Machine result](result.json)",
              "- [Server log](server/server.log)", "- [AgentX report](agentx/report/profile_export_aiperf.json)",
              "- [Shutdown](shutdown.json)", "- [Worker identities](identity-checks.json)", ""]
    (root / "result.md").write_text("\n".join(lines))
    with exclusive_lock(root.parent / ".summary.lock", blocking=True):
        summary = root.parent / "summary.md"
        new = not summary.exists()
        with summary.open("a") as target:
            if new:
                target.write("| Run | Outcome | Serving | Result |\n| --- | --- | --- | --- |\n")
            sha = manifest.get("sources", {}).get("serving", {}).get("commit", "unknown")
            target.write(f"| {root.name} | {result['outcome']} | `{sha}` | [report]({root.name}/result.md) |\n")


def collect_generated(manifest):
    """Retain default build outputs without forcing recompilation through an env override."""
    root = Path(manifest["run_dir"])
    copied = []
    for name in ("serving", "lib", "pypto", "runtime"):
        item = manifest.get("sources", {}).get(name)
        if not item:
            continue
        checkout = Path(item["path"])
        for source in checkout.glob("build_output*"):
            if source.is_dir() and not source.is_symlink():
                destination = root / "jit/generated" / name / source.name
                shutil.copytree(source, destination, dirs_exist_ok=True)
                copied.append({"source": str(source), "copy": str(destination.relative_to(root))})
    files = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size}
             for p in sorted((root / "jit").rglob("*")) if p.is_file()]
    write_json(root / "generated-artifacts.json", {"verified": True, "copied": copied, "files": files})


def run(args, profile):
    base = args.work_root.resolve() / "artifacts/dspark-regression"
    base.mkdir(parents=True, exist_ok=True)
    root = args.run_dir.resolve() if args.run_dir else Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=base))
    if args.run_dir:
        root.mkdir(parents=True, exist_ok=False)
    for name in ("environment/workers", "server/ascend", "agentx/report", "jit", "ci-sources"):
        (root / name).mkdir(parents=True, exist_ok=True)
    print(f"Regression artifacts: {root}", flush=True)
    manifest = {"schema_version": 1, "run_dir": str(root), "profile": profile, "port": args.port}
    write_json(root / "run.json", {"schema_version": 1, "profile": profile, "mode": "latest-ci", "at": now()})
    locks = ExitStack()
    previous_handlers = {}

    def interrupted(signum, _frame):
        raise RegressionError("INCOMPLETE", f"Controller interrupted by signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, interrupted)
    try:
        locks.enter_context(exclusive_lock(args.work_root.resolve() / ".deepseek-agentx.lock"))
        prefix, conda_sh, conda_exe = provision.conda_environment(args.conda_env)
        key = hashlib.sha256(str(prefix).encode()).hexdigest()
        locks.enter_context(exclusive_lock(Path.home() / ".cache/deepseek-agentx/locks" / (key + ".lock")))
        busy = provision.environment_busy(prefix)
        if busy:
            raise RegressionError("SKIPPED_BUSY", f"Environment is in use by processes {busy}")
        for name in ("git", "uv", "curl", "task-submit"):
            if not shutil.which(name):
                raise RegressionError("BLOCKED_SETUP", f"Missing executable: {name}")
        ensure_port_free(args.port)
        cann = args.cann_root.resolve() / "set_env.sh"
        if not cann.is_file():
            raise RegressionError("BLOCKED_SETUP", f"Missing CANN environment: {cann}")
        manifest.update(conda_prefix=str(prefix), conda_sh=str(conda_sh), conda_executable=str(conda_exe),
                        cann_env=str(cann), model=provision.model_identity(args.model.resolve()),
                        tokenizer=provision.tokenizer_identity((args.tokenizer or args.model).resolve()),
                        index_url=args.index_url)
        event(root, "preflight_complete")
        manifest.update(sources.resolve(root, profile))
        driver = root / "driver"
        (driver / "scripts").mkdir(parents=True)
        shutil.copy2(ROOT / "scripts/run_deepseek_agentx_regression.py", driver / "scripts")
        shutil.copytree(ROOT / "scripts/deepseek_agentx", driver / "scripts/deepseek_agentx",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        manifest["driver_root"] = str(driver)
        manifest["runner"] = {"source": str(ROOT), "commit": sources.git(ROOT, "rev-parse", "HEAD"),
                               "files": {str(p.relative_to(driver)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in driver.rglob("*") if p.is_file()}}
        write_json(root / "manifest.json", manifest)
        event(root, "sources_frozen", pypto=manifest["sources"]["pypto"]["commit"])
        provision.prepare(manifest)
        write_json(root / "environment/source-before.json", {"verified": True, "sources": sources.snapshot_sources(manifest)})
        server_script, _ = write_scripts(manifest)
        command = "bash " + shlex.quote(str(server_script))
        write_json(root / "server-submission-intent.json", {"command": command})
        server_task = tasks.submit(root, "server", command, 16, profile["budgets"])
        scanner = assessment.FaultScanner(root)
        wait_for(root, server_task, manifest, scanner, "server")
        with observe_traffic(root, args.port, profile["budgets"]["client_seconds"]) as proxy_port:
            _, client_script = write_scripts(manifest, proxy_port)
            command = "bash " + shlex.quote(str(client_script))
            write_json(root / "agentx-submission-intent.json", {"command": command})
            client_task = tasks.submit(root, "agentx", command, None, profile["budgets"])
            wait_for(root, client_task, manifest, scanner, "agentx", server_task)
        write_json(root / "finish-reason.json", {"outcome": "COMPLETED", "at": now()})
    except BaseException as exc:
        (root / "controller-error.log").write_text(traceback.format_exc())
        outcome = exc.outcome if isinstance(exc, RegressionError) else "BLOCKED_SETUP"
        write_json(root / "finish-reason.json", {"outcome": outcome, "at": now(),
                                                  "detail": f"{type(exc).__name__}: {exc}"})
        event(root, "stopping", outcome=outcome, detail=str(exc))
    finally:
        # Do not let a second cancellation interrupt cleanup; CI also has an always() fallback.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            cleanup(root, args.port,
                    profile["budgets"]["cleanup_seconds"])
            if (root / "environment/built.json").is_file():
                try:
                    record = sources.snapshot_sources(manifest)
                    env = provision.activation_environment(manifest)
                    run_command([Path(manifest["conda_prefix"]) / "bin/python",
                                 Path(manifest["driver_root"]) / "scripts/run_deepseek_agentx_regression.py", "identity", "--run-dir", root,
                                 "--output", root / "environment/after.json", "--with-imports"],
                                root / "environment/after.log", cwd=root, env=env, timeout=180)
                    before, after = read_json(root / "environment/built.json"), read_json(root / "environment/after.json")
                    if (before["imports"] != after["imports"] or before["runtime_libraries"] != after["runtime_libraries"]
                            or provision.model_identity(args.model.resolve()) != manifest["model"]
                            or provision.tokenizer_identity((args.tokenizer or args.model).resolve()) != manifest["tokenizer"]):
                        raise ValueError("Installed binaries, model or tokenizer changed during the run")
                    write_json(root / "environment/source-after.json", {"verified": True, "sources": record})
                except Exception as exc:
                    write_json(root / "environment/source-after.json", {"verified": False, "error": str(exc)})
            try:
                collect_generated(manifest)
            except (OSError, ValueError) as exc:
                write_json(root / "generated-artifacts.json", {"verified": False, "error": str(exc)})
            result = assessment.assess(root)
            render(root, manifest, result)
            print(f"Outcome: {result['outcome']}; report: {root / 'result.md'}", flush=True)
        finally:
            locks.close()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    return 0 if result["outcome"] == "PASS_STABILITY" else 1
