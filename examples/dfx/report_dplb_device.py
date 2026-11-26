# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Join a DSpark DFX capture with device timing and paired short-run service results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from pypto_serving.observability.dplb_dspark_timing import (
    ATTENTION_KINDS,
    semantic_scope_sites,
    summarize_dispatch,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def compare_metrics(case: Path, finished: int, tokens: int) -> dict:
    path = case / "metrics.jsonl"
    if not path.exists():
        return {}
    snapshots = [snapshot["replicas"] for snapshot in read_events(path)]

    def find_snapshot(requests, generated):
        return next(
            (
                replicas
                for replicas in reversed(snapshots)
                if sum(replica["counters"]["requests_finished"] for replica in replicas) == requests
                and sum(replica["counters"]["generation_tokens"] for replica in replicas) == generated
            ),
            None,
        )

    # The paired harness completes one single-token warmup before the capture.
    before = find_snapshot(1, 1)
    after = find_snapshot(1 + finished, 1 + tokens)
    if before is None or after is None:
        return {}
    result = {}
    for name in ("ttft", "itl"):
        sums = {}
        for field in ("sum", "count"):
            sums[field] = sum(replica["histograms"][name][field] for replica in after) - sum(
                replica["histograms"][name][field] for replica in before
            )
        result[f"{name}_seconds_sum"] = sums["sum"]
        result[f"{name}_intervals"] = sums["count"]
        result[f"{name}_mean_ms"] = round(1000 * sums["sum"] / sums["count"], 3) if sums["count"] else None
    for counter in ("preemptions", "prefix_cache_hits", "prefix_cache_query_tokens", "generation_tokens"):
        result[counter] = sum(replica["counters"][counter] for replica in after) - sum(
            replica["counters"][counter] for replica in before
        )
    return result


def summarize_service(case: Path) -> dict:
    path = case / "service.json"
    if not path.exists():
        return {}
    service = read_json(path)
    requests = service["requests"]
    tokens = sum(request["usage"]["completion_tokens"] for request in requests)
    seconds = service["wall_seconds"]
    first_chunks = [
        request["first_nonempty_chunk_seconds"]
        for request in requests
        if request["first_nonempty_chunk_seconds"] is not None
    ]
    return {
        **service,
        "output_tokens": tokens,
        "output_tokens_per_second": round(tokens / seconds, 4),
        "mean_first_nonempty_chunk_ms": round(mean(first_chunks) * 1000, 3) if first_chunks else None,
        "server_metrics": compare_metrics(case, len(requests), tokens),
    }


def timing_for_step(directory: Path | None) -> dict[int, dict]:
    if directory is None:
        return {}
    sources = list((directory / "source").glob("*.cpp"))
    if len(sources) != 1:
        return {}
    source = sources[0]
    sites = semantic_scope_sites(source)
    ranks = {}
    runs = set()
    for dispatch in sorted(directory.glob("rank[0-9]*/d[0-9]*")):
        program_marker = dispatch / "dispatch_program.timing.json"
        if not program_marker.is_file():
            program_marker = dispatch / "dispatch_program.json"
        if read_json(program_marker)["program"] != source.stem:
            raise ValueError(f"compiled orchestration differs from timed program in {dispatch}")
        timing = summarize_dispatch(dispatch, sites)
        rank = timing["rank"]
        if rank in ranks or dispatch.parent.name != f"rank{rank}":
            raise ValueError(f"ambiguous rank identity in {directory}")
        ranks[rank] = timing
        runs.add((timing["owner_instance_id"], timing["run_epoch"]))
    if len(runs) > 1:
        raise ValueError(f"rank records belong to different timed launches in {directory}")
    return ranks


def max_rank(ranks: list[dict], metric: str) -> float | None:
    values = [rank.get(metric) for rank in ranks]
    return max(values) if values and all(value is not None for value in values) else None


def decode_work(plan: dict, outcome: dict, invocation: dict) -> dict:
    captured = invocation.get("prepared_decode") or outcome.get("prepared_work") or {}
    if captured.get("phase") == "decode":
        return captured

    # Older mixed steps only reported prefill extents. A returned decode
    # result confirms execution even when the scheduler discarded it.
    candidates = [request for request in plan["requests"] if request.get("stage") == "decode"]
    returned = {request["request_id"]: request for request in outcome.get("requests", [])}
    executed = [
        request
        for request in candidates
        if returned.get(request["request_id"], {}).get("returned_tokens", 0) > 0
    ]
    return {
        "status": "worker_results_only" if executed else "unavailable",
        "request_rows": [
            {"request_id": request["request_id"], "group_id": request["cache_partition"]}
            for request in executed
        ],
        "group_rows": [
            {"group_id": group["group_id"], "valid_rows": 0}
            for group in plan["groups"]
            if not any(request["cache_partition"] == group["group_id"] for request in candidates)
        ],
    }


def group_row(plan: dict, prepared: dict, ranks: dict[int, dict], request_outcomes: dict[str, dict]) -> dict:
    group_id = plan["group_id"]
    ids = plan["tp_ranks"]
    valid_by_rank = prepared.get("valid_rows_by_rank") or []
    rank_rows = [
        {
            **ranks.get(rank, {"rank": rank, "status": "unavailable"}),
            "valid_rows": valid_by_rank[rank] if rank < len(valid_by_rank) else None,
        }
        for rank in ids
    ]
    rows = []
    for row in prepared.get("request_rows", []):
        if row["group_id"] != group_id:
            continue
        outcome = request_outcomes.get(row["request_id"], {})
        rows.append(
            {
                **row,
                **{
                    field: outcome.get(field)
                    for field in (
                        "request_uid",
                        "admission_epoch",
                        "returned_tokens",
                        "speculative_proposed_tokens",
                        "speculative_accepted_tokens",
                        "result_disposition",
                    )
                },
            }
        )
    extents = next((row for row in prepared.get("group_rows", []) if row["group_id"] == group_id), {})
    valid = extents.get("valid_rows")
    attention = max_rank(rank_rows, "attention_busy_us")
    state = "active" if rows or (valid or 0) > 0 else "empty" if valid == 0 else "unknown"
    consistency = None
    if attention is not None:
        if state == "empty" and attention > 0:
            consistency = "attention ran despite zero prepared rows"
        elif state == "active" and attention == 0:
            consistency = "attention missing despite executed decode work"
    attention_values = [rank.get("attention_busy_us") for rank in rank_rows]
    timed_records = sum(rank.get("records", 0) for rank in rank_rows)
    wait_matches = sum(rank.get("dispatch_to_start_matched_records", 0) for rank in rank_rows)
    return {
        "group_id": group_id,
        "state": state,
        "prepared_rows": extents,
        "prepared_requests": rows,
        "planned_request_ids": plan["participating_request_ids"],
        "planned_logical_tokens": plan["participating_logical_tokens"],
        "attention_kind_max_rank_us": {
            kind: max_rank([rank.get("attention_kind_us", {}) for rank in rank_rows], kind)
            for kind in ATTENTION_KINDS
        },
        "attention_max_rank_us": attention,
        "attention_mean_rank_us": round(mean(attention_values), 3) if attention is not None else None,
        "attention_rank_skew_us": round(attention - min(attention_values), 3)
        if attention is not None
        else None,
        "dispatch_to_start_max_rank_us": max_rank(rank_rows, "dispatch_to_start_us"),
        "dispatch_wait_matches": wait_matches,
        "timed_records": timed_records,
        "receive_to_start_max_rank_us": max_rank(rank_rows, "receive_to_start_us"),
        "in_window_idle_max_rank_us": max_rank(rank_rows, "in_window_idle_us"),
        "device_window_max_rank_us": max_rank(rank_rows, "device_window_us"),
        "consistency_warning": consistency,
        "ranks": rank_rows,
    }


def analyze_case(case: Path) -> dict:
    manifests = sorted((case / "profile" / "dplb").glob("*/*.manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"expected one serving capture in {case}; found {len(manifests)}")
    manifest_path = manifests[0]
    manifest = read_json(manifest_path)
    events = read_events(manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".jsonl")))
    if any(event["capture_id"] != manifest["capture_id"] for event in events):
        raise ValueError(f"mismatched serving capture IDs in {case}")
    placements = [event for event in events if event["event"] == "placement_commit"]
    outcomes = {event["invocation_id"]: event for event in events if event["event"] == "step_outcome"}
    device = {}
    for marker in (case / "device").glob("step-*/invocation.json"):
        invocation = read_json(marker)
        key = invocation["invocation_id"]
        if key in device:
            raise ValueError(f"more than one retained capture for {key}")
        device[key] = (marker.parent, invocation)
    trace_path = case / "profile/trace.json"
    trace = read_json(trace_path).get("traceEvents", []) if trace_path.exists() else []
    trace_keys = {
        event.get("args", {}).get("invocation_id")
        for event in trace
        if event.get("name") == "dspark_decode_one_l2.worker_submit"
    }
    steps = []
    for plan in (event for event in events if event["event"] == "step_plan"):
        key = plan["invocation_id"]
        outcome = outcomes.get(key) or {}
        if key not in trace_keys and key not in device and plan.get("stage") != "decode":
            continue
        directory, invocation = device.get(key, (None, None))
        if invocation is not None and any(
            invocation[field] != plan[field] for field in ("step_id", "execution_slot", "pipeline_generation")
        ):
            raise ValueError(f"device and serving records disagree on invocation {key}")
        error = None
        try:
            ranks = timing_for_step(directory)
            if ranks and set(ranks) != {rank for group in plan["groups"] for rank in group["tp_ranks"]}:
                raise ValueError("timed ranks differ from the serving topology")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            ranks = {}
            error = str(exc)
        if not ranks and error is None:
            error = "missing device capture or compiled orchestration for this step"
        prepared = decode_work(plan, outcome, invocation or {})
        prefill = outcome.get("prepared_work") or {}
        request_outcomes = {row["request_id"]: row for row in outcome.get("requests", [])}
        groups = [group_row(group, prepared, ranks, request_outcomes) for group in plan["groups"]]
        active = [group for group in groups if group["state"] == "active"]
        comparable = (
            key in trace_keys
            and active
            and all(
                group["state"] != "unknown"
                and not group["consistency_warning"]
                and group["attention_max_rank_us"] is not None
                for group in groups
            )
            and all(
                group["attention_max_rank_us"] is not None and group["attention_max_rank_us"] > 0
                for group in active
            )
        )
        slowest = max(active, key=lambda group: group["attention_max_rank_us"]) if comparable else None
        request_ids = {row["request_id"] for row in prepared.get("request_rows", [])}
        steps.append(
            {
                "step": plan["step_id"],
                "invocation_id": key,
                "host_profile_joined": key in trace_keys,
                "prepared_status": prepared.get("status", "unavailable"),
                "stage": plan["stage"],
                "prefill_work": prefill if prefill.get("phase") == "prefill" else None,
                "device_identity_source": invocation.get("identity_source", "submission")
                if invocation
                else None,
                "placement_requests": sorted(request_ids),
                "groups": groups,
                "device_error": error,
                "slowest_group": slowest["group_id"] if slowest else None,
                "slowest_attention_us": slowest["attention_max_rank_us"] if slowest else None,
            }
        )
    service = summarize_service(case)
    participating = {
        (row["request_id"], row["request_uid"], row["admission_epoch"])
        for step in steps
        for group in step["groups"]
        for row in group["prepared_requests"]
        if row["request_uid"] is not None
    }
    placed = {(p["request_id"], p["request_uid"], p["admission_epoch"]) for p in placements}
    completed = {request["request_id"] for request in service.get("requests", [])}
    return {
        "case": case.name,
        "capture_id": manifest["capture_id"],
        "capture_complete": manifest.get("complete"),
        "runtime_coverage": manifest.get("runtime_coverage", {}).get("verdict"),
        "placement_to_load_to_service_ids": sorted({key[0] for key in participating & placed} & completed),
        "placements": placements,
        "steps": steps,
        "service": service,
    }


def display(value) -> str:
    return "—" if value is None else str(value)


def observed_peak(case: dict) -> float | None:
    values = [step["slowest_attention_us"] for step in case["steps"]]
    return max(values) if values and all(value is not None for value in values) else None


def compare_cases(cases: list[dict]) -> dict:
    by_mode = {case["case"]: case for case in cases}
    if set(by_mode) != {"off", "on"} or len(cases) != 2:
        return {}
    off, on = (by_mode[mode] for mode in ("off", "on"))

    def workload(case: dict) -> list:
        return sorted(
            (record["usage"]["prompt_tokens"], record["usage"]["completion_tokens"])
            for record in case["service"].get("requests", [])
        )

    if not workload(off) or workload(off) != workload(on):
        return {"status": "unavailable", "reason": "unequal or missing completed workload"}
    if not all(case["capture_complete"] for case in cases):
        return {"status": "unavailable", "reason": "incomplete serving capture"}
    off_peak, on_peak = observed_peak(off), observed_peak(on)
    if not off_peak or not on_peak:
        return {"status": "unavailable", "reason": "at least one step lacks a comparable slowest group"}
    return {
        "status": "observed",
        "off_peak_slowest_group_us": off_peak,
        "on_peak_slowest_group_us": on_peak,
        "change_percent": round(100 * (on_peak / off_peak - 1), 2),
    }


def markdown(cases: list[dict], comparison: dict) -> str:
    lines = [
        "# DSpark DPLB device timing and service metrics",
        "",
        "Times are in µs. Overlaps are merged within each rank, then each DP group takes the maximum rank duration. "
        "Attention timings include all timed operators in the executed branch. Empty refers to zero Decode work. "
        "Dispatch wait uses matched same-rank tasks. Idle is the uncovered time within the observed AICore window.",
        "",
        "| DPLB | Completed | Output tokens | Mean first nonempty chunk ms | Server TTFT ms | "
        "Server ITL ms | Client token/s | Peak slowest group µs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        service = case["service"]
        server = service.get("server_metrics", {})
        lines.append(
            "| "
            + " | ".join(
                map(
                    display,
                    (
                        case["case"],
                        len(service["requests"]) if service else None,
                        service.get("output_tokens"),
                        service.get("mean_first_nonempty_chunk_ms"),
                        server.get("ttft_mean_ms"),
                        server.get("itl_mean_ms"),
                        service.get("output_tokens_per_second"),
                        observed_peak(case),
                    ),
                )
            )
            + " |"
        )
    if comparison.get("status") == "observed":
        lines.extend(
            [
                "",
                f"DPLB off/on peak slowest-group Attention was {comparison['off_peak_slowest_group_us']:.3f} µs "
                f"and {comparison['on_peak_slowest_group_us']:.3f} µs "
                f"({comparison['change_percent']:+.2f}%) in this paired short run. The service metrics "
                "describe the same requests with device capture enabled.",
            ]
        )
    elif comparison:
        lines.extend(["", f"The paired comparison is unavailable because {comparison['reason']}."])
    for case in cases:
        lines.extend(
            [
                "",
                f"## DPLB {case['case']}",
                "",
                f"Serving capture complete: {case['capture_complete']}. "
                f"{sum(step['slowest_attention_us'] is not None for step in case['steps'])}/{len(case['steps'])} "
                f"steps comparable. {len(case['placement_to_load_to_service_ids'])} requests join to "
                "their service responses.",
                "",
                "| Request | Selected group | Reason | Prompt tokens | Prefix hits by group |",
                "| --- | ---: | --- | ---: | --- |",
            ]
        )
        for place in case["placements"]:
            hits = ", ".join(
                f"{group['group_id']}={display(group.get('prefix_hit_tokens'))}" for group in place["groups"]
            )
            lines.append(
                f"| {place['request_id']} | {place.get('selected_group')} | {place.get('reason')} | "
                f"{place.get('prompt_tokens')} | {hits} |"
            )
        if server := case["service"].get("server_metrics"):
            lines.extend(
                [
                    "",
                    f"The serving window has {server['ttft_intervals']} TTFT samples, "
                    f"{server['itl_intervals']} token intervals, {server['prefix_cache_hits']} prefix hit "
                    f"tokens and {server['preemptions']} preemptions.",
                ]
            )
        recovered = [
            str(step["step"])
            for step in case["steps"]
            if step["device_identity_source"] == "host_submit_window"
        ]
        if recovered:
            lines.extend(
                [
                    "",
                    f"Step identities for {', '.join(recovered)} were recovered from host submit windows "
                    "and raw device records.",
                ]
            )
        if any(step["prepared_status"] == "worker_results_only" for step in case["steps"]):
            lines.extend(
                [
                    "",
                    "For older mixed steps, a worker result confirms Decode execution. "
                    "Decode extents are unavailable. See prefill_work in the JSON for Prefill extents.",
                ]
            )
        lines.extend(
            [
                "",
                "| Step | DP group | Executed requests | Valid / physical rows | SWA | HCA | CSA | "
                "Attention | Dispatch wait | Wait matches | Device idle | Device window |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for step in case["steps"]:
            for group in step["groups"]:
                kinds = group["attention_kind_max_rank_us"]
                if group["state"] == "unknown":
                    requests = "—"
                else:
                    requests = ", ".join(row["request_id"] for row in group["prepared_requests"]) or "Empty"
                extents = group["prepared_rows"]
                rows = f"{display(extents.get('valid_rows'))}/{display(extents.get('physical_rows'))}"
                lines.append(
                    "| "
                    + " | ".join(
                        map(
                            display,
                            (
                                step["step"],
                                group["group_id"],
                                requests,
                                rows,
                                kinds["swa"],
                                kinds["hca"],
                                kinds["csa"],
                                group["attention_max_rank_us"],
                                group["dispatch_to_start_max_rank_us"],
                                f"{group['dispatch_wait_matches']}/{group['timed_records']}"
                                if group["timed_records"]
                                else None,
                                group["in_window_idle_max_rank_us"],
                                group["device_window_max_rank_us"],
                            ),
                        )
                    )
                    + " |"
                )
            if step["device_error"]:
                lines.append(f"\nStep {step['step']} device timing unavailable. {step['device_error']}\n")
        lines.extend(
            [
                "",
                "| Step | DP group | Rank | Valid rows | SWA | HCA | CSA | Attention | Dispatch wait | Device idle |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for step in case["steps"]:
            for group in step["groups"]:
                for rank in group["ranks"]:
                    kinds = rank.get("attention_kind_us", {})
                    lines.append(
                        "| "
                        + " | ".join(
                            map(
                                display,
                                (
                                    step["step"],
                                    group["group_id"],
                                    rank["rank"],
                                    rank["valid_rows"],
                                    kinds.get("swa"),
                                    kinds.get("hca"),
                                    kinds.get("csa"),
                                    rank.get("attention_busy_us"),
                                    rank.get("dispatch_to_start_us"),
                                    rank.get("in_window_idle_us"),
                                ),
                            )
                        )
                        + " |"
                    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = [analyze_case(case) for case in args.case]
    comparison = compare_cases(cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "cases": cases, "comparison": comparison}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(markdown(cases, comparison))


if __name__ == "__main__":
    main()
