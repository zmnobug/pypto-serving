# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explain existing repair runs without modifying their workspaces or trajectories."""

import argparse
from collections import deque
import json
from pathlib import Path
import re

from failure_signals import verification_signature

EDIT_COMMAND = re.compile(r"\bsed\s+-i\b|\bwrite_text\s*\(|\bapply_patch\b")


def trajectory_actions(messages: list[dict]) -> list[dict]:
    """Recover proposed actions and any saved outputs from legacy runs."""
    pending = deque()
    actions = []
    for message_index, message in enumerate(messages):
        extra = message.get("extra", {})
        for action in extra.get("actions", []):
            index = len(actions)
            actions.append({
                "index": index, "assistant_message_index": message_index,
                "observation_message_index": None, "command": action.get("command", ""),
                "returncode": None, "changed_files": None, "stalled": False,
                "verification_passed": None, "observed": False,
            })
            pending.append(index)
        if "raw_output" not in extra or not pending:
            continue
        action = actions[pending.popleft()]
        match = re.search(r"<returncode>(-?\d+)</returncode>", message.get("content", ""))
        action.update({
            "observation_message_index": message_index,
            "returncode": int(match.group(1)) if match else None,
            "changed_files": extra.get("changed_files"),
            "stalled": bool(extra.get("stalled")),
            "verification_passed": extra.get("verification", {}).get("passed"),
            "observed": True,
        })
    return actions


def analyze_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    if run_dir.is_file():
        run_dir = run_dir.parent
    summary = json.loads((run_dir / "summary.json").read_text())
    trajectory_path = run_dir / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.is_file() else {}
    messages = trajectory.get("messages", [])
    context = trajectory.get("info", {}).get("context_management", {}).get("requests", [])
    requests_path = run_dir / "request_events.json"
    requests = json.loads(requests_path.read_text()) if requests_path.is_file() else []
    actions_path = run_dir / "action_events.json"
    actions = json.loads(actions_path.read_text()) if actions_path.is_file() else trajectory_actions(messages)
    findings = []

    def add(kind, *, message_index=None, action_index=None, request_index=None, detail=None):
        finding = {"kind": kind}
        for key, value in (("message_index", message_index), ("action_index", action_index),
                           ("request_index", request_index), ("detail", detail)):
            if value is not None:
                finding[key] = value
        findings.append(finding)

    previous_check = None
    for message_index, message in enumerate(messages):
        extra = message.get("extra", {})
        if extra.get("interrupt_type") == "FormatError":
            add("format_error", message_index=message_index, detail=extra.get("finish_reason"))
        verification = extra.get("verification")
        explicit = False
        if verification is None and isinstance(extra.get("raw_output"), str):
            try:
                decoded, _ = json.JSONDecoder().raw_decode(extra["raw_output"].lstrip())
            except ValueError:
                decoded = None
            if isinstance(decoded, dict) and decoded.get("passed") is False and (
                "public" in decoded or "independent" in decoded
            ):
                verification = {
                    "passed": False, "output": json.dumps(decoded),
                    "reasons": ["Verification command failed with exit code 1."],
                }
                explicit = True
        if not verification or verification.get("passed"):
            continue
        signature = verification_signature(verification)
        add("explicit_failed_check" if explicit else "failed_check", message_index=message_index, detail=signature)
        if previous_check and signature == previous_check["signature"]:
            add("repeated_failed_check", message_index=message_index, detail={
                "previous_message_index": previous_check["message_index"],
                "tracked_files_changed": verification.get("current_files") != previous_check["files"],
            })
        previous_check = {"signature": signature, "message_index": message_index,
                          "files": verification.get("current_files")}

    for action in actions:
        index = action["index"]
        message_index = action.get("observation_message_index")
        if action.get("stalled"):
            add("stalled_action", action_index=index, message_index=message_index)
        if action.get("changed_files") == [] and EDIT_COMMAND.search(action.get("command", "")):
            add("possible_no_op_edit", action_index=index, message_index=message_index)

    for index, event in enumerate(context):
        if event.get("dropped_message_indices") or event.get("shortened_message_indices"):
            add("context_trimmed", request_index=index, detail={
                "prompt_tokens": event.get("prompt_tokens"),
                "output_reserve": event.get("output_reserve"),
            })
        if event.get("history_restarted"):
            add("history_restarted", request_index=index,
                detail=event.get("history_restart_reason"))
    for index, event in enumerate(requests):
        if event.get("status") == "error":
            add("request_error", request_index=index, detail=event.get("error_type"))
        if event.get("finish_reason") == "length":
            add("output_truncated", request_index=index)
    if not requests:
        for index, message in enumerate(messages):
            response = message.get("extra", {}).get("response")
            if isinstance(response, dict) and response.get("choices", [{}])[0].get("finish_reason") == "length":
                add("output_truncated", message_index=index)

    counts = {kind: sum(item["kind"] == kind for item in findings) for kind in sorted({
        item["kind"] for item in findings
    })}
    elapsed = summary.get("elapsed_seconds")
    model_time = summary.get("model_seconds")
    command_time = summary.get("command_seconds_including_gate")
    return {
        "run_dir": str(run_dir), "variant": summary.get("variant"),
        "transport": summary.get("transport"), "accepted": summary.get("accepted"),
        "exit_status": summary.get("result", {}).get("exit_status"),
        "fixture_sha256": summary.get("fixture_sha256"),
        "responses": summary.get("responses"), "verification_attempts": summary.get("verification_attempts"),
        "time_seconds": {
            "task": elapsed, "model_query": model_time,
            "inference": summary.get("inference_seconds"),
            "metrics_sampling": summary.get("serving_sampling_seconds"),
            "commands_including_gate": command_time,
            "other_within_task": max(0, elapsed - model_time - command_time)
            if all(isinstance(value, (int, float)) for value in (elapsed, model_time, command_time)) else None,
        },
        "action_source": "action_events.json" if actions_path.is_file() else "trajectory",
        "request_source": "request_events.json" if requests_path.is_file() else "trajectory",
        "action_count": len(actions),
        "observed_action_count": sum(action.get("observed", True) for action in actions),
        "request_count": len(requests) or summary.get("responses"),
        "finding_counts": counts, "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run folders, summary.json files, or suite folders")
    parser.add_argument("--output", type=Path, help="Write a separate JSON report; never modify input runs")
    args = parser.parse_args()
    paths = []
    for path in args.runs:
        if path.is_file() or (path / "summary.json").is_file():
            paths.append(path)
        else:
            paths.extend(sorted(path.rglob("summary.json")))
    if not paths:
        parser.error("No summary.json files found")
    report = []
    for path in paths:
        summary_path = path if path.is_file() else path / "summary.json"
        summary = json.loads(summary_path.read_text())
        if isinstance(summary, dict) and isinstance(summary.get("result"), dict):
            report.append(analyze_run(path))
    if not report:
        parser.error("No Agent experiment summaries found")
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(result + "\n")
        print(args.output)
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
