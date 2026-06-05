# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import python.profile.recorder as recorder_module
from python.profile.env import ProfileConfig, load_profile_config
from python.profile.merge import merge_fragments
from python.profile.recorder import ProfileRecorder, _prepare_run, get_profiler


def test_profile_config_disabled_by_default():
    config = load_profile_config({})

    assert config.enabled is False
    assert config.output == Path("./profile_out")
    assert config.trace_file == Path("./profile_out/trace.json")
    assert config.fragments_dir == Path("./profile_out/fragments")
    assert config.levels == frozenset({"e2e", "kernel"})


def test_profile_config_level_env_enables_default_output():
    config = load_profile_config({"SA_PROFILE_LEVEL": "verbose"})

    assert config.enabled is True
    assert config.trace_file == Path("./profile_out/trace.json")
    assert config.fragments_dir == Path("./profile_out/fragments")
    assert config.levels == frozenset({"e2e", "kernel", "verbose"})
    assert config.includes("e2e")
    assert config.includes("kernel")


def test_profile_config_json_output_uses_sibling_fragment_dir(tmp_path):
    output = tmp_path / "trace.json"

    config = load_profile_config({"SA_PROFILE_OUTPUT": str(output)})

    assert config.enabled is True
    assert config.output == output
    assert config.trace_file == output
    assert config.fragments_dir == tmp_path / "trace.json.fragments"


def test_profile_recorder_writes_mergeable_chrome_trace(tmp_path):
    config = ProfileConfig(
        enabled=True,
        output=tmp_path,
        trace_file=tmp_path / "trace.json",
        fragments_dir=tmp_path / "fragments",
        levels=frozenset({"e2e", "kernel"}),
    )

    recorder = ProfileRecorder(config, process_name="unit-process")
    with recorder.span("unit.span", cat="unit", args={"value": 1}):
        recorder.instant("unit.instant", cat="unit", args={"state": "inside"})
        recorder.duration("unit.duration", cat="unit", dur_us=1234)
    recorder.close()

    count = merge_fragments(config.fragments_dir, config.trace_file)
    trace = json.loads(config.trace_file.read_text())
    events = trace["traceEvents"]

    assert count == len(events)
    assert any(event["ph"] == "M" and event["name"] == "process_name" for event in events)
    assert any(event["ph"] == "M" and event["name"] == "thread_name" for event in events)
    assert any(event["ph"] == "i" and event["name"] == "unit.instant" for event in events)

    span = next(event for event in events if event["ph"] == "X" and event["name"] == "unit.span")
    assert span["cat"] == "unit"
    assert span["dur"] >= 0
    assert span["args"] == {"value": 1}

    duration = next(event for event in events if event["ph"] == "X" and event["name"] == "unit.duration")
    assert duration["dur"] == 1234
    assert duration["args"] == {}


def test_prepare_run_removes_stale_fragments(tmp_path):
    config = ProfileConfig(
        enabled=True,
        output=tmp_path,
        trace_file=tmp_path / "trace.json",
        fragments_dir=tmp_path / "fragments",
        levels=frozenset({"e2e"}),
    )
    config.fragments_dir.mkdir()
    stale = config.fragments_dir / "trace.123.jsonl"
    stale.write_text("{}\n")
    unrelated = config.fragments_dir / "notes.txt"
    unrelated.write_text("keep")

    _prepare_run(config)

    assert not stale.exists()
    assert unrelated.exists()


def test_merge_fragments_ignores_malformed_lines(tmp_path):
    fragments_dir = tmp_path / "fragments"
    fragments_dir.mkdir()
    fragment = fragments_dir / "trace.1.jsonl"
    fragment.write_text('{"name":"ok","ph":"i"}\n{"name":\n{"name":"ok2","ph":"i"}\n')
    trace_file = tmp_path / "trace.json"

    count = merge_fragments(fragments_dir, trace_file)
    events = json.loads(trace_file.read_text())["traceEvents"]

    assert count == 2
    assert [event["name"] for event in events] == ["ok", "ok2"]


def test_get_profiler_updates_process_name_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("SA_PROFILE_OUTPUT", str(tmp_path))
    monkeypatch.delenv("SA_PROFILE_MAIN_PID", raising=False)
    recorder_module._profiler = None
    recorder_module._profiler_pid = None

    profiler = get_profiler(process_name="initial")
    profiler = get_profiler(process_name="renamed")
    profiler.close()

    merge_fragments(profiler.config.fragments_dir, profiler.config.trace_file)
    events = json.loads(profiler.config.trace_file.read_text())["traceEvents"]
    process_names = [
        event["args"]["name"]
        for event in events
        if event["ph"] == "M" and event["name"] == "process_name"
    ]

    assert process_names[-2:] == ["initial", "renamed"]
    recorder_module._profiler = None
    recorder_module._profiler_pid = None


def test_profile_recorder_uses_async_task_lanes(tmp_path):
    config = ProfileConfig(
        enabled=True,
        output=tmp_path,
        trace_file=tmp_path / "trace.json",
        fragments_dir=tmp_path / "fragments",
        levels=frozenset({"e2e"}),
    )
    recorder = ProfileRecorder(config, process_name="unit-process")

    async def record_in_task() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.set_name("unit-task")
        with recorder.span("async.span"):
            recorder.instant("async.instant")

    asyncio.run(record_in_task())
    recorder.close()

    merge_fragments(config.fragments_dir, config.trace_file)
    events = json.loads(config.trace_file.read_text())["traceEvents"]

    assert any(
        event["ph"] == "M"
        and event["name"] == "thread_name"
        and event["args"]["name"] == "unit-task"
        for event in events
    )
