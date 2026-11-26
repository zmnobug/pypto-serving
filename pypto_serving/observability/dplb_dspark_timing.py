# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Retain the timed DSpark device launch before the worker reuses its output directory."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from .dplb_device_timing import interval_union_cycles


def enable_single_pass_capture(config) -> None:
    """Use the pinned PyPTO L3 timing pass once for this decode config.

    The ordinary swimlane path executes the graph and timing passes on the
    same mutable KV buffers. This diagnostic consumes timing and scope files,
    so it skips the graph execution and online viewer conversion.
    """
    from pypto.runtime import distributed_runner

    run_default = distributed_runner._run_l3_swimlane_two_pass
    collect_default = distributed_runner._collect_program_swimlane
    pending = threading.local()

    def run_capture(dc, requested, dfx_base, run_pass):
        if requested is not config:
            return run_default(dc, requested, dfx_base, run_pass)
        call_config = distributed_runner._make_call_config(
            dc,
            requested,
            dfx_base=dfx_base,
            co_enable_swimlane_dep_gen=False,
        )
        run_pass(call_config)
        pending.raw = True

    def collect_capture(compiled):
        if getattr(pending, "raw", False):
            pending.raw = False
            return
        collect_default(compiled)

    distributed_runner._run_l3_swimlane_two_pass = run_capture
    distributed_runner._collect_program_swimlane = collect_capture


def _retain(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def retain_capture(root: Path, compiled, invocation, prepared_decode: dict | None = None) -> None:
    from pypto.runtime._artifact_runtime import runtime_output_directory

    runtime_root = Path(runtime_output_directory(compiled)) / "dfx_outputs"
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{invocation.step_id}-{invocation.pipeline_generation}"
    with tempfile.TemporaryDirectory(dir=root, prefix=f".{name}-") as staging:
        destination = Path(staging) / name
        destination.mkdir()
        _retain_artifacts(runtime_root, Path(compiled.output_dir), destination)
        (destination / "invocation.json").write_text(
            json.dumps(
                {
                    **invocation.as_profile_args(),
                    "prepared_decode": prepared_decode,
                }
            )
            + "\n"
        )
        destination.rename(root / name)


def _retain_artifacts(runtime_root: Path, compiled_root: Path, destination: Path) -> None:
    artifacts = (
        "chip_swimlane_records.json",
        "dispatch_identity.json",
        "dispatch_identity.timing.json",
        "dispatch_program.json",
        "dispatch_program.timing.json",
        "scope_stats/scope_stats.jsonl",
    )
    programs = set()
    for dispatch in sorted(runtime_root.glob("rank[0-9]*/d[0-9]*")):
        if not (dispatch / artifacts[0]).is_file():
            continue
        for filename in artifacts:
            candidate = dispatch / filename
            if candidate.is_file():
                _retain(candidate, destination / dispatch.relative_to(runtime_root) / filename)
        marker = dispatch / "dispatch_program.timing.json"
        if not marker.is_file():
            marker = dispatch / "dispatch_program.json"
        if marker.is_file():
            program = json.loads(marker.read_text()).get("program")
            if isinstance(program, str) and program.isidentifier():
                programs.add(program)
    for program in programs:
        source = compiled_root / "next_levels" / program / "orchestration" / f"{program}.cpp"
        if source.is_file():
            target = destination / "source" / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)


ATTENTION_KINDS = ("swa", "hca", "csa")


def semantic_scope_sites(source: Path) -> dict[str, str]:
    """Resolve scoped attention branches from this exact DSpark L2 build.

    The generated orchestration keeps the group-token guard and kernel
    comments. The guard excludes the interleaved MoE and repacking scopes.
    Refuse an unknown layout instead of using name prefixes to guess the
    semantic kind of shared kernels.
    """
    lines = source.read_text().splitlines()
    markers = {
        "swa": re.compile(r"// Spmd swa_merge_pack_publish_spmd\b|// Spmd swa_merge_pack_publish_spmd_\d+\b"),
        "hca": re.compile(r"// Group hca_raw_attn:\s"),
        "csa": re.compile(r"// Spmd csa_merge_pack_publish_spmd\b|// Spmd csa_merge_pack_publish_spmd_\d+\b"),
    }
    sites = {}
    found = Counter()
    for index, line in enumerate(lines[1:], 1):
        if "if ((0 < static_cast<int64_t>(group_tokens" not in line or not line.endswith(" {"):
            continue
        if lines[index - 1].strip() != "SIMPLER_SCOPE() {":
            continue
        indent = line[: len(line) - len(line.lstrip())]
        end = next((j for j in range(index + 1, len(lines)) if lines[j] == indent + "}"), None)
        if end is None:
            raise ValueError("unterminated DSpark attention branch")
        block = "\n".join(lines[index + 1 : end])
        kinds = [kind for kind, pattern in markers.items() if pattern.search(block)]
        if not kinds:
            continue
        if len(kinds) != 1:
            raise ValueError("ambiguous DSpark attention branch")
        kind = kinds[0]
        found[kind] += 1
        for offset in range(index - 1, end):
            if lines[offset].lstrip().startswith("SIMPLER_SCOPE("):
                sites[f"{source.name}:{offset + 1}"] = kind
    if found != {"swa": 2, "hca": 1, "csa": 2}:
        raise ValueError(f"unsupported DSpark attention scope layout: {dict(found)}")
    return sites


def _semantic_task_ranges(path: Path, epoch: int, sites: dict[str, str]) -> dict[int, list]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not records or records[0].get("fatal") is not False or records[0].get("dropped") != 0:
        raise ValueError("missing, incomplete or dropped runtime scope records")
    if records[0].get("total") != len(records) - 1:
        raise ValueError("scope record count differs from runtime metadata")
    open_scopes = []
    ranges = defaultdict(list)
    seen_sites = set()
    for record in records[1:]:
        if record.get("run_epoch") != epoch:
            raise ValueError("scope and AICore records carry different run epochs")
        phase = record["phase"]
        site = record["site"]
        ring = record["ring"]
        depth = record["depth"]
        head = record["task_window_end"]
        kind = sites.get(site)
        if kind is None and ":" in site:
            file, line = site.rsplit(":", 1)
            kind = sites.get(f"{Path(file).name}:{line}")
        identity = (site, ring, depth)
        if phase == "begin":
            open_scopes.append((identity, head))
            continue
        if phase != "end" or not open_scopes:
            raise ValueError("scope records are unbalanced")
        start_identity, first_task = open_scopes.pop()
        if identity != start_identity or first_task > head:
            raise ValueError("scope boundaries disagree on task window")
        if kind:
            seen_sites.add(kind)
            if first_task != head:
                ranges[ring].append((first_task, head, kind))
    if open_scopes or seen_sites != set(ATTENTION_KINDS):
        raise ValueError("timed launch lacks complete SWA/HCA/CSA scope boundaries")
    merged_ranges = {}
    for ring, spans in ranges.items():
        merged = []
        for start, end, kind in sorted(spans):
            if merged and start < merged[-1][1]:
                previous_start, previous_end, previous_kind = merged[-1]
                if kind != previous_kind:
                    raise ValueError("one device task belongs to multiple attention kinds")
                # Scope depth can exceed the number of rings. Both the parent and
                # child then observe the same task indices on the final ring.
                merged[-1] = (previous_start, max(previous_end, end), kind)
            else:
                merged.append((start, end, kind))
        merged_ranges[ring] = merged
    return merged_ranges


def summarize_dispatch(dispatch: Path, sites: dict[str, str]) -> dict:
    raw = json.loads((dispatch / "chip_swimlane_records.json").read_text())
    marker = dispatch / "dispatch_identity.timing.json"
    if not marker.is_file():
        marker = dispatch / "dispatch_identity.json"
    identity = json.loads(marker.read_text())
    rows = raw["aicore_tasks"]
    if not rows or raw["metadata"]["runtime"] != "tensormap_and_ringbuffer":
        raise ValueError("expected one timed DSpark task-ring launch")
    epochs = {row[6] for row in rows}
    if len(epochs) != 1 or identity.get("run_id") not in epochs or min(epochs) <= 0:
        raise ValueError("timed dispatch and AICore epochs disagree")
    epoch = epochs.pop()
    raw_frequency = raw["metadata"]["clock_freq_hz"]
    if type(raw_frequency) is not int or raw_frequency <= 0:
        raise ValueError("invalid device frequency")

    scope_error = None
    try:
        ranges = _semantic_task_ranges(dispatch / "scope_stats/scope_stats.jsonl", epoch, sites)
    except (OSError, ValueError, KeyError) as error:
        ranges = {}
        scope_error = str(error)
    range_starts = {ring: [span[0] for span in spans] for ring, spans in ranges.items()}
    times = {kind: [] for kind in ATTENTION_KINDS}
    seen = set()
    all_times = []
    waits = []
    received_waits = []
    scheduler_rows = raw.get("scheduler_tasks", {}).get("records", [])
    scheduled = {(row[4], row[0], row[1]): (row[2], row[3]) for row in scheduler_rows}
    if len(scheduled) != len(scheduler_rows):
        raise ValueError("ambiguous scheduler registration")
    scheduler_matches = 0
    for core, token, registration, start, end, receive_wait, epoch in rows:
        key = (epoch, core, registration)
        if key in seen or min(core, token, registration, receive_wait) < 0 or not 0 < start < end:
            raise ValueError("invalid or duplicate AICore record")
        seen.add(key)
        all_times.append((start, end))
        if 0 < receive_wait <= start:
            received_waits.append((start - receive_wait, start))
        scheduling = scheduled.get(key)
        if scheduling is not None:
            dispatched, finished = scheduling
            if 0 < dispatched <= start <= end <= finished:
                waits.append((dispatched, start))
                scheduler_matches += 1
        ring = token >> 32
        task_index = token & 0xFFFF_FFFF
        if ring in ranges:
            location = bisect_right(range_starts[ring], task_index) - 1
            if location >= 0:
                _, last, kind = ranges[ring][location]
                if task_index < last:
                    times[kind].append((start, end))

    def us(cycles: int) -> float:
        return round(cycles * 1_000_000 / raw_frequency, 3)

    attention_times = [interval for kind in ATTENTION_KINDS for interval in times[kind]]
    all_busy = interval_union_cycles(all_times)
    envelope = max(end for _, end in all_times) - min(start for start, _ in all_times)
    return {
        "rank": identity["chip_rank"],
        "owner_instance_id": identity["owner_instance_id"],
        "run_epoch": epoch,
        "clock_freq_hz": raw_frequency,
        "records": len(rows),
        "record_coverage": "observed",
        "scope_error": scope_error,
        "attention_kind_us": {
            kind: us(interval_union_cycles(times[kind])) if scope_error is None else None
            for kind in ATTENTION_KINDS
        },
        "attention_kind_aicore_records": {kind: len(value) for kind, value in times.items()},
        "attention_busy_us": us(interval_union_cycles(attention_times)) if scope_error is None else None,
        "device_window_us": us(envelope),
        "device_busy_us": us(all_busy),
        "in_window_idle_us": us(envelope - all_busy),
        "dispatch_to_start_us": us(interval_union_cycles(waits)) if scheduler_matches else None,
        "dispatch_to_start_matched_records": scheduler_matches,
        "dispatch_to_start_status": (
            "observed"
            if scheduler_matches == len(rows)
            else "partial"
            if scheduler_matches
            else "unavailable"
        ),
        "receive_to_start_us": us(interval_union_cycles(received_waits)),
    }
