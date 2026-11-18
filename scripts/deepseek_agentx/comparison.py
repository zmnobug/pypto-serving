# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare two regression manifests and outcomes without claiming causality."""

from __future__ import annotations

from pathlib import Path

from .common import read_json


SOURCE_NAMES = ("serving", "lib", "lib_ci", "pypto", "runtime", "agentx")
RESULT_FIELDS = ("outcome", "server_received", "server_finished", "actual_prefix_hits", "profiling_seconds",
                 "client_phases", "request_errors", "incomplete_reasons", "first_fault")


def _manifest(root):
    path = Path(root) / "manifest.json"
    return read_json(path)


def _result(root):
    path = Path(root) / "result.json"
    return read_json(path) if path.is_file() else {}


def _without_paths(value):
    if isinstance(value, dict):
        return {key: _without_paths(item) for key, item in value.items()
                if key not in {"path", "mtime_ns", "run_dir", "driver_root", "conda_prefix", "conda_sh",
                               "conda_executable"}}
    if isinstance(value, list):
        return [_without_paths(item) for item in value]
    return value


def _source_commits(manifest):
    return {name: manifest.get("sources", {}).get(name, {}).get("commit") for name in SOURCE_NAMES}


def _source_changes(baseline, candidate):
    changes = {}
    for name in SOURCE_NAMES:
        before, after = baseline.get(name), candidate.get(name)
        if before != after:
            changes[name] = {"baseline": before, "candidate": after}
    return changes


def _result_view(result):
    return {field: result.get(field) for field in RESULT_FIELDS}


def _fault_signature(result):
    fault = result.get("first_fault")
    if not isinstance(fault, dict):
        return None
    return {key: fault.get(key) for key in ("outcome", "file", "line", "text")}


def compare_runs(baseline_root, candidate_root):
    baseline_root, candidate_root = Path(baseline_root), Path(candidate_root)
    baseline_manifest, candidate_manifest = _manifest(baseline_root), _manifest(candidate_root)
    baseline_result, candidate_result = _result(baseline_root), _result(candidate_root)
    baseline_sources, candidate_sources = _source_commits(baseline_manifest), _source_commits(candidate_manifest)
    source_changes = _source_changes(baseline_sources, candidate_sources)
    baseline_profile = _without_paths(baseline_manifest.get("profile"))
    candidate_profile = _without_paths(candidate_manifest.get("profile"))
    baseline_model = _without_paths(baseline_manifest.get("model"))
    candidate_model = _without_paths(candidate_manifest.get("model"))
    baseline_tokenizer = _without_paths(baseline_manifest.get("tokenizer"))
    candidate_tokenizer = _without_paths(candidate_manifest.get("tokenizer"))
    baseline_result_view, candidate_result_view = _result_view(baseline_result), _result_view(candidate_result)
    result_changes = {key: {"baseline": baseline_result_view[key], "candidate": candidate_result_view[key]}
                      for key in RESULT_FIELDS if baseline_result_view[key] != candidate_result_view[key]}
    configuration_changes = {
        "workload": baseline_profile != candidate_profile,
        "model": baseline_model != candidate_model,
        "tokenizer": baseline_tokenizer != candidate_tokenizer,
        "toolchain": {
            key: {"baseline": _without_paths(baseline_manifest.get(key)),
                  "candidate": _without_paths(candidate_manifest.get(key))}
            for key in ("ci", "ptoas", "pto_isa")
            if _without_paths(baseline_manifest.get(key)) != _without_paths(candidate_manifest.get(key))
        },
    }
    behavior_changed = bool(result_changes)
    incomparable = (configuration_changes["workload"] or configuration_changes["model"]
                    or configuration_changes["tokenizer"])
    if incomparable:
        classification = "INCOMPARABLE_CONFIGURATION"
    elif source_changes and behavior_changed:
        classification = "BEHAVIOR_CHANGE_WITH_SOURCE_CHANGES"
    elif source_changes:
        classification = "SOURCE_CHANGE_NO_OBSERVED_RESULT_CHANGE"
    elif behavior_changed:
        classification = "BEHAVIOR_CHANGE_SAME_SOURCES"
    else:
        classification = "NO_OBSERVED_CHANGE"
    likely = []
    if len(source_changes) == 1 and behavior_changed and not incomparable:
        likely = list(source_changes)
    return {
        "schema_version": 1,
        "baseline_run": baseline_root.name,
        "candidate_run": candidate_root.name,
        "classification": classification,
        "source_commits": {"baseline": baseline_sources, "candidate": candidate_sources},
        "source_changes": source_changes,
        "configuration_changes": configuration_changes,
        "result_changes": result_changes,
        "first_fault": {"baseline": _fault_signature(baseline_result),
                         "candidate": _fault_signature(candidate_result)},
        "likely_changed_components": likely,
        "causality": ("A single source component changed while the observed result changed; this is a candidate for bisect, not proof of root cause."
                       if likely else "The comparison does not isolate a single causal component."),
    }


def render_comparison(output, comparison):
    output = Path(output)
    lines = ["# DeepSeek + AgentX comparison", "", f"Classification: **{comparison['classification']}**", "",
             f"Baseline: `{comparison['baseline_run']}`", f"Candidate: `{comparison['candidate_run']}`", "",
             "| Component | Baseline | Candidate |", "| --- | --- | --- |"]
    for name in SOURCE_NAMES:
        item = comparison["source_changes"].get(name)
        if item:
            lines.append(f"| {name} | `{item['baseline']}` | `{item['candidate']}` |")
    if not comparison["source_changes"]:
        lines.append("| source commits | unchanged | unchanged |")
    lines += ["", "## Result changes", ""]
    if comparison["result_changes"]:
        for key, item in comparison["result_changes"].items():
            lines.append(f"- `{key}`: `{item['baseline']}` → `{item['candidate']}`")
    else:
        lines.append("- No recorded result fields changed.")
    lines += ["", "## Interpretation", "", f"- {comparison['causality']}", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
