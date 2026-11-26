# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded task-submit lifecycle, restricted to this run's recorded tasks."""

import re
import subprocess
import time

from .common import RegressionError, capture, clean_environment, event, read_json, write_json

TASK_ID = re.compile(r"task_\d{8}_\d{6}_\d+")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def status(task):
    if not TASK_ID.fullmatch(task):
        raise ValueError("Invalid task id")
    process = subprocess.run(["task-submit", "--status", task], capture_output=True, text=True,
                             timeout=30, env=clean_environment())
    text = ANSI.sub("", process.stdout + process.stderr).strip()
    if process.returncode:
        return {"state": "unknown", "exit_code": None, "text": text}
    if re.fullmatch(r"completed \(exit=(-?\d+)\)", text):
        return {"state": "completed", "exit_code": int(re.search(r"exit=(-?\d+)", text)[1]), "text": text}
    if text in {"pending", "running"}:
        return {"state": text, "exit_code": None, "text": text}
    return {"state": "unknown", "exit_code": None, "text": text}


def submit(root, role, command, devices, budgets):
    args = ["task-submit"]
    args += ["--device", "auto", "--device-num", str(devices)] if devices else ["--no-device"]
    args += ["--timeout", str(budgets["queue_seconds"]), "--max-time",
             str(budgets["server_seconds"] if role == "server" else budgets["client_seconds"]), command]
    process = subprocess.run(args, capture_output=True, text=True, timeout=60, env=clean_environment())
    output = process.stdout + process.stderr
    (root / f"{role}-submission.log").write_text(output)
    ids = TASK_ID.findall(ANSI.sub("", output))
    if process.returncode or len(set(ids)) != 1:
        raise RegressionError("BLOCKED_QUEUE", f"Task submission failed; see {role}-submission.log")
    task = ids[0]
    write_json(root / f"{role}-task.json", {"task_id": task, "command": command})
    event(root, f"{role}_submitted", task_id=task)
    return task


def stop(root, role, budget):
    path = root / f"{role}-task.json"
    if not path.is_file():
        return {"state": "not_submitted", "verified": True, "exit_code": None}
    task = read_json(path)["task_id"]
    current = status(task)
    if current["state"] == "completed":
        unexpected = role == "server"
        return {**current, "task_id": task, "verified": not unexpected, "stopped_by_controller": False,
                "unexpected_completion": unexpected}
    cancelled = current["state"] == "pending"
    action = "--cancel" if cancelled else "--kill"
    result = subprocess.run(["task-submit", action, task], capture_output=True, text=True,
                            timeout=30, env=clean_environment())
    (root / f"{role}-stop.log").write_text(result.stdout + result.stderr)
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        current = status(task)
        if current["state"] == "completed":
            return {**current, "task_id": task, "verified": True, "stopped_by_controller": True}
        if cancelled and result.returncode == 0 and current["state"] == "unknown":
            return {"state": "cancelled", "verified": True, "task_id": task,
                    "exit_code": None, "stopped_by_controller": True}
        if cancelled and current["state"] == "running":
            # The queue can start a task between status inspection and cancellation.
            cancelled = False
            result = subprocess.run(["task-submit", "--kill", task], capture_output=True, text=True,
                                    timeout=30, env=clean_environment())
            with (root / f"{role}-stop.log").open("a") as output:
                output.write(result.stdout + result.stderr)
        time.sleep(1)
    return {**current, "task_id": task, "verified": False, "stopped_by_controller": True}


def recover_submission(root, role):
    """Recover a submission interrupted between queue acceptance and recording its ID."""
    if (root / f"{role}-task.json").exists() or not (root / f"{role}-submission-intent.json").exists():
        return
    command = read_json(root / f"{role}-submission-intent.json")["command"]
    found = capture(["task-submit", "--find", command], timeout=30, env=clean_environment())
    ids = [line for line in found.splitlines() if TASK_ID.fullmatch(line)]
    if len(ids) == 1:
        write_json(root / f"{role}-task.json", {"task_id": ids[0], "command": command, "recovered": True})
    elif ids:
        raise RegressionError("INCOMPLETE", "More than one task matches this run's unique command")
