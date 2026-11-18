# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared artifact, process and configuration primitives."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "tests/regression/deepseek_agentx/full.json"


class RegressionError(Exception):
    def __init__(self, outcome: str, detail: str):
        super().__init__(detail)
        self.outcome = outcome


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event(root, kind, **values):
    record = {"at": now(), "event": kind, **values}
    with (Path(root) / "events.jsonl").open("a") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"[{record['at']}] {kind}: {values}", flush=True)


def capture(args, *, cwd=None, env=None, timeout=120):
    return subprocess.check_output(
        list(map(str, args)), cwd=cwd, env=env, stderr=subprocess.STDOUT, timeout=timeout, text=True,
    ).strip()


def git(root, *args):
    return capture(["git", "-C", root, *args])


def run_command(args, log, *, cwd=None, env=None, timeout=7200):
    """Bounded command with a heartbeat and cleanup of its own process group."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {args[0]} (log: {log})", flush=True)
    with log.open("a") as output:
        output.write(f"\n{now()} {list(map(str, args))!r}\n")
        output.flush()
        process = subprocess.Popen(
            list(map(str, args)), cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline, heartbeat = time.monotonic() + timeout, 0.0
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise RegressionError("BLOCKED_SETUP", f"Command timed out; see {log}")
                if time.monotonic() >= heartbeat:
                    print(f"Still running; see {log}", flush=True)
                    heartbeat = time.monotonic() + 30
                time.sleep(0.2)
            if process.returncode:
                raise RegressionError("BLOCKED_SETUP", f"Command exited {process.returncode}; see {log}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


@contextmanager
def exclusive_lock(path, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RegressionError("SKIPPED_BUSY", f"Lock held: {path}") from exc
        yield


def validate_profile(profile):
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported regression profile schema")
    budgets = profile["budgets"]
    if budgets["client_seconds"] < profile["agentx"]["duration_seconds"] + 2400:
        raise ValueError("Client budget must cover configuration, warmup, load and report generation")
    if budgets["server_seconds"] < (budgets["startup_seconds"] + budgets["queue_seconds"]
                                     + budgets["client_seconds"] + budgets["cleanup_seconds"]
                                     + budgets["fault_flush_seconds"]):
        raise ValueError("Server budget does not cover startup, client queue, client execution and cleanup")
    if profile["server"]["devices"] != 16 or not profile["server"]["prefix_cache"]:
        raise ValueError("The full regression requires 16 devices and prefix caching")
    for value in budgets.values():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("All budgets must be positive integers")
    server = profile["server"]
    if (server["dp"], server["tp"], server["ep"], server["speculative_tokens"]) != (4, 4, 16, 7):
        raise ValueError("The full profile requires DP4/TP4/EP16 and DSpark K7")
    if profile["agentx"]["duration_seconds"] < 1800 or profile["agentx"]["concurrency"] != 4:
        raise ValueError("The full profile requires at least 1800 formal seconds and concurrency 4")
    return profile


def load_profile(path=PROFILE):
    return validate_profile(read_json(path))


def clean_environment():
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("PYPTO_", "PTOAS_", "PTO_ISA_", "SIMPLER_", "ASCEND_GLOBAL_LOG")) or key in {
            "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_CONSTRAINT", "UV_CONSTRAINT",
            "TASKQUEUE_IGNORE_WHITELIST", "TASKQUEUE_IGNORE_GROUP_AFFINITY",
        }:
            env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    return env
