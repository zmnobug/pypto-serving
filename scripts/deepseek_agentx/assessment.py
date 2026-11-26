# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Conservative offline assessment; missing evidence never becomes zero errors."""

from collections import Counter
import json
import math
from pathlib import Path
import re

from .common import now, read_json, write_json
from .identity import KEYS

FAULT = re.compile(
    r"\b507018\b|DDR address of the MTE instruction is out of range|"
    r"ACL_ERROR_RT_AICORE_EXCEPTION|ACL_ERROR_RT_AICPU_EXCEPTION|"
    r"AICore task exception|AICPU task exception|\bMTE.*(?:out of range|exception)",
    re.IGNORECASE,
)
HOST_FAULT = re.compile(r"Traceback \(most recent call last\)|non-finite logits|illegal token|out of memory|"
                        r"Worker (?:process|step) failed|DSpark packed prefill dispatch failed",
                        re.IGNORECASE)


def number(value):
    if isinstance(value, dict):
        value = value.get("avg", value.get("value"))
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def phase_duration(rows, phase):
    timestamps = []
    for row in rows:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            return None
        if metadata.get("benchmark_phase") != phase:
            continue
        start = metadata.get("request_start_ns")
        end = metadata.get("request_end_ns")
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool) or end < start):
            return None
        timestamps.append((start, end))
    if not timestamps:
        return None
    return (max(end for _, end in timestamps) - min(start for start, _ in timestamps)) / 1_000_000_000


class FaultScanner:
    def __init__(self, root):
        self.root, self.offsets = Path(root), {}

    def scan(self):
        paths = [self.root / "server/server.log"]
        paths += sorted(p for p in (self.root / "server/ascend").rglob("*") if p.is_file())
        for path in paths:
            if not path.is_file():
                continue
            stat = path.stat()
            key = (str(path), stat.st_ino)
            offset, lineno = self.offsets.get(key, (0, 0))
            if stat.st_size < offset:
                offset, lineno = 0, 0
            with path.open("rb") as source:
                source.seek(offset)
                for raw in source:
                    lineno += 1
                    line = raw.decode(errors="replace")
                    if raw.endswith(b"\n"):
                        self.offsets[key] = (source.tell(), lineno)
                    match = FAULT.search(line)
                    host = path == self.root / "server/server.log" and HOST_FAULT.search(line)
                    if match or host:
                        return {"file": str(path.relative_to(self.root)), "line": lineno,
                                "text": line.strip()[:2000], "outcome": "FAIL_DEVICE" if match else "FAIL_SERVER"}
        return None


def assess(root):
    root = Path(root)
    missing, errors = [], []

    def read(name, default=None):
        try:
            value = read_json(root / name)
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            return value
        except (OSError, ValueError):
            missing.append("Missing or invalid " + name)
            return {} if default is None else default

    def records(name):
        try:
            values = [json.loads(line) for line in (root / name).read_text().splitlines() if line.strip()]
            if not values or any(not isinstance(item, dict) for item in values):
                raise ValueError("Expected nonempty object records")
            return values
        except (OSError, ValueError):
            missing.append("Missing or invalid " + name)
            return []

    config = read("run.json")
    profile = config.get("profile", {})
    if not isinstance(profile, dict):
        profile = {}
        missing.append("Invalid workload profile")
    client = profile.get("agentx", {})
    if not isinstance(client, dict):
        client = {}
        missing.append("Invalid AgentX profile")
    report = read("agentx/report/profile_export_aiperf.json")
    if not isinstance(report, dict):
        report = {}
        missing.append("AgentX report must be an object")
    rows = records("agentx/report/profile_export.jsonl")
    traffic = records("replay/requests.jsonl")
    valid_traffic = []
    for item in traffic:
        if not isinstance(item.get("client_request_id"), str) or not isinstance(item.get("server_request_id"), str):
            missing.append("Missing HTTP client/server request identity")
        else:
            valid_traffic.append(item)
    traffic = valid_traffic
    try:
        log = (root / "server/server.log").read_text(errors="replace")
    except OSError:
        log = ""
        missing.append("Missing server log")
    received = re.findall(r"request (\S+) received: prompt=(\d+)", log)
    finished = re.findall(r"request (\S+) finished: prompt=(\d+) out=(\d+) reason=(\S+)", log)
    received_ids = Counter(x[0] for x in received)
    finished_ids = Counter(x[0] for x in finished)
    if not received or received_ids != finished_ids or any(v != 1 for v in received_ids.values()):
        missing.append("Server received/finished IDs do not join uniquely")
    finished_map = {item[0]: item for item in finished}
    for request, _, output, reason in finished:
        if reason not in {"FINISHED_LENGTH", "FINISHED_STOP", "FINISHED_EOS"} or int(output) <= 0:
            errors.append(f"Server request failed or empty: {request}: {reason}")
    hit_ids = re.findall(r"prefix_cache_hit request_id=(\S+) prefix_cache_hit_tokens=([0-9]+)", log)
    hits = [item for item in hit_ids if int(item[1]) > 0 and item[0] in received_ids]
    if not hits:
        missing.append("No actual positive prefix-cache hit joined to this run")
    phases = Counter()
    client_ids = Counter()
    client_map = {}
    for row in rows:
        metadata, metrics = row.get("metadata"), row.get("metrics")
        if not isinstance(metadata, dict) or not isinstance(metrics, dict):
            missing.append("Unrecognized AgentX request schema")
            continue
        phase = metadata.get("benchmark_phase")
        if not isinstance(phase, str) or phase not in {"warmup", "profiling"}:
            missing.append("Unknown AgentX benchmark phase")
            continue
        phases[phase] += 1
        request = metadata.get("x_request_id")
        if not isinstance(request, str) or not request:
            missing.append("Missing AgentX request identity")
            continue
        client_ids[request] += 1
        client_map[request] = row
        if metadata.get("was_cancelled") is not False or metadata.get("context_overflow_skip") is not False:
            missing.append("Per-request cancellation/skip status not explicitly false")
        if row.get("error") or metadata.get("error"):
            errors.append(f"AgentX request error: {request}")
        output = number(metrics.get("usage_completion_tokens"))
        if output is None:
            missing.append(f"Missing output usage: {request}")
        elif output <= 0:
            errors.append(f"Empty AgentX output: {request}")
    observed_client = Counter(item.get("client_request_id") for item in traffic)
    observed_server = Counter(item.get("server_request_id") for item in traffic)
    if not client_ids or client_ids != observed_client or any(v != 1 for v in client_ids.values()):
        missing.append("Client/HTTP observer IDs do not join uniquely")
    if received_ids != observed_server:
        missing.append("HTTP observer/server IDs do not join uniquely")
    for item in traffic:
        request = item.get("client_request_id")
        if item.get("error") or item.get("status") != 200 or item.get("finish_reason") not in {"stop", "length"}:
            errors.append(f"HTTP/SSE request failed: {request}")
        if item.get("done") is not True:
            missing.append(f"Missing SSE DONE: {request}")
        usage = item.get("usage")
        server = finished_map.get(item.get("server_request_id"))
        row = client_map.get(request)
        if not isinstance(usage, dict) or not server or not row:
            missing.append(f"Missing joined usage evidence: {request}")
            continue
        metrics = row["metrics"]
        if any(number(value) is None for value in (usage.get("prompt_tokens"), usage.get("completion_tokens"),
                                                   metrics.get("usage_prompt_tokens"),
                                                   metrics.get("usage_completion_tokens"))):
            missing.append(f"Missing finite joined token counts: {request}")
            continue
        if (number(usage.get("completion_tokens")) != number(metrics.get("usage_completion_tokens"))
                or number(usage.get("prompt_tokens")) != number(metrics.get("usage_prompt_tokens"))
                or number(usage.get("completion_tokens")) != int(server[2])
                or number(usage.get("prompt_tokens")) != int(server[1])):
            errors.append(f"Client/HTTP/server usage mismatch: {request}")
    if report.get("schema_version") != "1.4":
        missing.append("Unvalidated AgentX report schema (expected 1.4)")
    reported_duration = number(report.get("benchmark_duration"))
    if reported_duration is None:
        missing.append("AgentX aggregate duration unavailable")
    if isinstance(report.get("benchmark_duration"), dict) and report["benchmark_duration"].get("unit", "sec") not in {
        "sec", "s", "seconds",
    }:
        missing.append("AgentX duration unit is not seconds")
    duration = phase_duration(rows, "profiling")
    if duration is None or duration < client.get("duration_seconds", 1800):
        missing.append("AgentX profiling duration shorter than required or unknown")
    if (phases["warmup"] != client.get("warmup_requests", 4) or not phases["profiling"]
            or number(report.get("request_count")) != phases["profiling"]):
        missing.append("Warmup/profiling counts incomplete")
    if report.get("was_cancelled") is not False or not isinstance(report.get("error_summary"), list):
        missing.append("AgentX final cancellation/error summary unavailable")
    rate = number(report.get("request_error_rate"))
    if rate is None:
        missing.append("AgentX final error rate unavailable")
    elif rate != 0 or report.get("error_summary"):
        errors.append("AgentX reports request errors")
    identities = read("identity-checks.json").get("checks", {})
    if not isinstance(identities, dict):
        identities = {}
    if any(identities.get(key) is not True for key in KEYS):
        missing.append("Actual model-worker identities not fully verified")
    after = read("environment/source-after.json")
    if after.get("verified") is not True:
        missing.append("Final source/native/model identity not verified")
    generated = read("generated-artifacts.json")
    generated_files = generated.get("files")
    if generated.get("verified") is not True or not isinstance(generated_files, list) or not generated_files:
        missing.append("Generated kernel/cache archive not verified")
    shutdown = read("shutdown.json")
    if shutdown.get("verified") is not True:
        missing.append("Task, port or process cleanup not verified")
    shutdown_tasks = shutdown.get("tasks", {})
    server_task = shutdown_tasks.get("server", {}) if isinstance(shutdown_tasks, dict) else {}
    server_failure = server_task.get("unexpected_completion") is True
    if server_failure:
        missing.append("Server task exited before controller cleanup")
    task = read("agentx-final.json")
    if task.get("state") != "completed" or task.get("exit_code") != 0 or task.get("natural") is not True:
        missing.append("AgentX did not finish naturally with exit zero")
    cann_files = [p for p in (root / "server/ascend").rglob("*") if p.is_file() and p.stat().st_size]
    if not cann_files:
        missing.append("No nonempty CANN logs")
    fault = FaultScanner(root).scan()
    finish = read("finish-reason.json")
    reason = finish.get("outcome", "INCOMPLETE")
    if fault:
        outcome = fault["outcome"]
    elif server_failure:
        outcome = "FAIL_SERVER"
    elif reason != "COMPLETED":
        known = {"FAIL_DEVICE", "FAIL_SERVER", "FAIL_REQUESTS", "INCOMPLETE", "SKIPPED_BUSY",
                 "BLOCKED_SETUP", "BLOCKED_QUEUE", "BLOCKED_FETCH", "BLOCKED_COMPATIBILITY"}
        outcome = reason if reason in known else "INCOMPLETE"
    elif errors:
        outcome = "FAIL_REQUESTS"
    elif missing:
        outcome = "INCOMPLETE"
    else:
        outcome = "PASS_STABILITY"
    result = {"schema_version": 1, "at": now(), "outcome": outcome,
        "correctness": "NOT_RUN", "performance_comparison": "NOT_RUN",
        "stability": outcome, "cleanup": "PASS" if shutdown.get("verified") else "INCOMPLETE",
        "server_received": len(received), "server_finished": len(finished),
        "actual_prefix_hits": len(hits), "client_phases": dict(phases), "profiling_seconds": duration,
        "first_fault": fault, "finish_reason": finish, "request_errors": sorted(set(errors)),
        "incomplete_reasons": sorted(set(missing)), "numeric_equivalence_tested": False}
    write_json(root / "result.json", result)
    return result
