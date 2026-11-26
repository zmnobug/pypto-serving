# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test timing attribution using scopes from the timed run."""

import json
from types import SimpleNamespace

import pytest

from pypto_serving.config.types import DispatchInvocation
from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin
from pypto_serving.observability.dplb_dspark_timing import (
    enable_single_pass_capture,
    semantic_scope_sites,
    summarize_dispatch,
)


def _scope_layout():
    markers = (
        "Spmd swa_merge_pack_publish_spmd: swa_merge_pack_publish",
        "Spmd swa_merge_pack_publish_spmd_0: swa_merge_pack_publish_0",
        "Spmd csa_merge_pack_publish_spmd: csa_merge_pack_publish",
        "Group hca_raw_attn: MixedKernels",
        "Spmd csa_merge_pack_publish_spmd_0: csa_merge_pack_publish_0",
    )
    return "\n".join(
        "    SIMPLER_SCOPE() {\n"
        "        if ((0 < static_cast<int64_t>(group_tokens))) {\n"
        f"            // {marker}\n"
        "            SIMPLER_SCOPE() {\n"
        "                rt_submit_aiv_task(1, params);\n"
        "            }\n"
        "        }\n"
        "    }"
        for marker in markers
    )


@pytest.fixture
def capture(tmp_path):
    source = tmp_path / "decode.cpp"
    source.write_text(_scope_layout())
    sites = semantic_scope_sites(source)
    records = []
    tag_ranges = ((0, 3), (3, 3), (3, 4), (4, 5), (5, 5))
    all_sites = list(sites)

    def add(site, phase, depth, head):
        records.append(dict(site=site, phase=phase, ring=3, depth=depth, task_window_end=head, run_epoch=7))

    for index, (start, end) in enumerate(tag_ranges):
        parent, child = all_sites[index * 2 : index * 2 + 2]
        add(parent, "begin", 3, start)
        if end > start:
            add(child, "begin", 4, start)
            add(child, "end", 4, end)
        add(parent, "end", 3, end)
    scope = tmp_path / "scope_stats/scope_stats.jsonl"
    scope.parent.mkdir()
    scope.write_text("\n".join(map(json.dumps, [dict(fatal=False, dropped=0, total=len(records)), *records])))
    rows = [
        [0, (3 << 32) + index, index, start, end, 2, 7]
        for index, (start, end) in enumerate(((10, 30), (20, 40), (35, 45), (40, 55), (60, 70), (80, 100)))
    ]
    raw = dict(
        metadata=dict(runtime="tensormap_and_ringbuffer", clock_freq_hz=1_000_000),
        aicore_tasks=rows,
        scheduler_tasks=dict(
            records=[[core, seq, start - 3, end + 1, epoch] for core, _, seq, start, end, _, epoch in rows]
        ),
    )
    (tmp_path / "chip_swimlane_records.json").write_text(json.dumps(raw))
    (tmp_path / "dispatch_identity.json").write_text(json.dumps(dict(run_id=6)))
    (tmp_path / "dispatch_identity.timing.json").write_text(
        json.dumps(dict(run_id=7, chip_rank=0, owner_instance_id="owner"))
    )
    return tmp_path, sites, scope


def test_timing_joins_same_run_scopes_and_partitions_overlapping_spans(capture):
    path, sites, _ = capture
    timing = summarize_dispatch(path, sites)
    assert timing["attention_kind_us"] == {"swa": 35.0, "hca": 10.0, "csa": 15.0}
    assert timing["attention_busy_us"] == 55.0
    assert timing["device_busy_us"] == 75.0
    assert timing["in_window_idle_us"] == 15.0
    assert timing["dispatch_to_start_us"] == 18.0
    assert timing["dispatch_to_start_status"] == "observed"


def test_other_run_or_dropped_scope_is_unavailable(capture):
    path, sites, scope = capture
    original = scope.read_text()
    for broken in (
        original.replace('"run_epoch": 7', '"run_epoch": 6', 1),
        original.replace('"dropped": 0', '"dropped": 1', 1),
    ):
        scope.write_text(broken)
        timing = summarize_dispatch(path, sites)
        assert timing["scope_error"]
        assert timing["attention_kind_us"] == {"swa": None, "hca": None, "csa": None}
        assert timing["device_busy_us"] == 75.0


def test_l3_submit_retains_its_invocation_if_next_step_is_prepared_during_submit():
    launch = DispatchInvocation("case:9", 9, 1, 5)
    next_launch = DispatchInvocation("case:10", 10, 0, 4)

    class Runner(L3DispatchMixin):
        _l3_run_config = None

        def _shared_l3_worker(self):
            return self

        def submit(self, compiled, *, config):
            self.set_invocation_context(next_launch)
            return object()

    runner = Runner()
    runner.set_invocation_context(launch)
    spec = SimpleNamespace(
        name="decode", compiled=object(), dispatch_args=(), block_dim=None, aicpu_thread_num=4
    )
    pending = runner._submit_l3(spec)
    assert pending.invocation == launch
    assert pending.profile_args["invocation_id"] == launch.invocation_id


def test_raw_capture_executes_mutable_decode_once(monkeypatch):
    from pypto.runtime import distributed_runner

    diagnostic_config = object()
    usual_config = object()
    actions = []

    def ordinary_pass(dc, config, dfx_base, run_pass):
        run_pass("graph")
        run_pass("timing")

    def make_config(dc, config, *, dfx_base, co_enable_swimlane_dep_gen):
        assert not co_enable_swimlane_dep_gen
        return "raw timing"

    monkeypatch.setattr(distributed_runner, "_run_l3_swimlane_two_pass", ordinary_pass)
    monkeypatch.setattr(distributed_runner, "_make_call_config", make_config)
    monkeypatch.setattr(distributed_runner, "_collect_program_swimlane", lambda _: actions.append("convert"))
    enable_single_pass_capture(diagnostic_config)

    for config in (diagnostic_config, usual_config):
        distributed_runner._run_l3_swimlane_two_pass(None, config, None, actions.append)
        distributed_runner._collect_program_swimlane(None)

    assert actions == ["raw timing", "graph", "timing", "convert"]
