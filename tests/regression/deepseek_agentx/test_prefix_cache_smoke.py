# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU evidence and lifecycle guards for the two-request DSpark NPU smoke."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import test_deepseek_dspark_accuracy as guard  # noqa: E402


@pytest.fixture
def pair():
    responses = [
        {"id": name, "model": guard.MODEL_ID,
         "usage": {"prompt_tokens": count, "completion_tokens": 16, "total_tokens": count + 16},
         "choices": [{"text": "A generated answer.", "finish_reason": "length"}]}
        for name, count in (("cmpl-producer", 512), ("cmpl-consumer", 530))
    ]
    log = "\n".join(
        f"request {response['id']} received: prompt={response['usage']['prompt_tokens']} tokens\n"
        f"request {response['id']} finished: prompt={response['usage']['prompt_tokens']} "
        "out=16 reason=FINISHED_LENGTH e2e=1.0s"
        for response in responses
    ) + "\nprefix_cache_hit request_id=cmpl-consumer prefix_cache_hit_tokens=384\n"
    return responses, log


def test_consumer_cache_hit_is_joined_to_its_own_completed_request(pair):
    responses, log = pair
    assert guard._assert_prefix_cache_pair(log, *responses)["cached_tokens"] == 384


@pytest.mark.parametrize("old,new", [
    ("request_id=cmpl-consumer", "request_id=another-request"),
    ("prefix_cache_hit_tokens=384", "prefix_cache_hit_tokens=0"),
    ("prefix_cache_hit_tokens=384", "prefix_cache_hit_tokens=129"),
    ("prefix_cache_hit_tokens=384", "prefix_cache_hit_tokens=512"),
    ("cmpl-consumer finished", "another-request finished"),
    ("out=16 reason=FINISHED_LENGTH", "out=15 reason=FINISHED_LENGTH"),
    ("reason=FINISHED_LENGTH", "reason=ABORTED"),
])
def test_unrelated_hits_bad_boundaries_and_bad_completions_cannot_pass(pair, old, new):
    responses, log = pair
    with pytest.raises(AssertionError):
        guard._assert_prefix_cache_pair(log.replace(old, new), *responses)


@pytest.mark.parametrize("extra", [
    "prefix_cache_hit request_id=cmpl-producer prefix_cache_hit_tokens=128",
    "prefix_cache_hit request_id=cmpl-consumer prefix_cache_hit_tokens=384",
    "request cmpl-consumer received: prompt=530 tokens",
])
def test_preseeded_or_ambiguous_evidence_cannot_pass(pair, extra):
    responses, log = pair
    with pytest.raises(AssertionError):
        guard._assert_prefix_cache_pair(log + extra + "\n", *responses)


def test_natural_eos_with_nonempty_output_is_a_valid_completion(pair):
    responses, log = pair
    responses = copy.deepcopy(responses)
    responses[1]["choices"][0]["finish_reason"] = "stop"
    responses[1]["usage"].update(completion_tokens=3, total_tokens=533)
    log = log.replace("prompt=530 out=16 reason=FINISHED_LENGTH", "prompt=530 out=3 reason=FINISHED_EOS")
    guard._assert_prefix_cache_pair(log, *responses)


@pytest.mark.parametrize("fail_consumer", [False, True])
def test_minimal_runner_keeps_evidence_and_stops_its_server(tmp_path, monkeypatch, pair, fail_consumer):
    responses, log = pair
    calls, stopped, launches = [], [], []
    monkeypatch.setenv("PYPTO_DSV4_DSPARK_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(guard, "_task_devices", lambda: tuple(range(16)))
    monkeypatch.setattr(guard, "_wait_for_health", lambda *_: None)
    monkeypatch.setattr(guard, "_wait_for_device_reclaim", lambda *_: None)
    monkeypatch.setattr(guard, "_stop_process_group", lambda process: stopped.append(process))

    def launch(command, **kwargs):
        launches.append((command, kwargs["env"]))
        kwargs["stdout"].write(log + guard.ONE_L2_COMPILE_MARKER + "\nDSpark speculation progress\n")
        kwargs["stdout"].flush()
        return object()

    def request(*_args, **kwargs):
        calls.append(kwargs["prompt"])
        if len(calls) == 2 and fail_consumer:
            raise RuntimeError("consumer failed")
        return responses[len(calls) - 1]

    monkeypatch.setattr(guard.subprocess, "Popen", launch)
    monkeypatch.setattr(guard, "_request_completion", request)
    if fail_consumer:
        with pytest.raises(RuntimeError, match="consumer failed"):
            guard.test_dspark_http_prefix_cache_minimal(tmp_path)
    else:
        guard.test_dspark_http_prefix_cache_minimal(tmp_path)
    result = json.loads((tmp_path / "prefix-cache-minimal-result.json").read_text())
    assert calls == [guard.PREFIX_CACHE_MINIMAL_CASE.prompt, guard.PREFIX_CACHE_MINIMAL_CASE.reuse_prompt]
    assert calls[0] != calls[1]
    assert len(stopped) == 1
    assert result["outcome"] == ("FAIL" if fail_consumer else "PASS_PREFIX_CACHE_SMOKE")
    assert len(result["requests"]) == (1 if fail_consumer else 2)
    assert result["numerical_equivalence"] == "NOT_RUN"
    command, env = launches[0]
    assert "--enable-prefix-caching" in command
    assert json.loads(command[command.index("--speculative-config") + 1])["num_speculative_tokens"] == 7
    assert env["ASCEND_PROCESS_LOG_PATH"] == str(tmp_path / "ascend")
