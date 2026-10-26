# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3 tool calls through the Chat HTTP and engine delivery path."""

import json

import pytest
from fastapi.testclient import TestClient

from pypto_serving.serving.reasoning.qwen3_tools import TOOL_END, TOOL_START
from tests.unit.serving.reasoning.test_qwen3_tools import QwenToolTokenizer
from tests.unit.serving.server.test_tool_calls import TOOLS, _assert_released, _collect_chat, _replay_server


class _QwenChatTokenizer(QwenToolTokenizer):
    output_parser_id = "qwen3"
    eos_token_id = QwenToolTokenizer.vocab["<eos>"]
    bos_token_id = None

    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append((messages, kwargs))
        return "Question"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 100000])
def test_tool_roundtrip_and_followup_answer(stream, chunk_size):
    call = TOOL_START + '{"name":"lookup","arguments":{"city":"杭州"}}' + TOOL_END + "<eos>"
    server = _replay_server(
        (call, "FINISHED_EOS"), ("Done<eos>", "FINISHED_EOS"),
        chunk_size=chunk_size, tokenizer=_QwenChatTokenizer(),
    )
    messages = [{"role": "user", "content": "Find city"}]
    with TestClient(server.app) as client:
        first, reason, _ = _collect_chat(client.post("/v1/chat/completions", json={
            "messages": messages, "tools": TOOLS, "stream": stream,
        }), stream)
        assert reason == "tool_calls"
        assert first["content"] is None
        assert json.loads(first["tool_calls"][0]["function"]["arguments"]) == {"city": "杭州"}
        messages += [first, {"role": "tool", "tool_call_id": first["tool_calls"][0]["id"], "content": "Found"}]
        second, reason, _ = _collect_chat(client.post("/v1/chat/completions", json={
            "messages": messages, "tools": TOOLS, "stream": stream,
        }), stream)
    assert second["content"] == "Done"
    assert reason == "stop"
    assert server.engine.tokenizer.prompts[1][0][1]["tool_calls"][0]["id"] == first["tool_calls"][0]["id"]
    _assert_released(server, 2)


@pytest.mark.parametrize("stream", [False, True])
def test_tool_free_qwen_chat_keeps_existing_unparsed_output(stream):
    raw = "<think>hidden</think>Answer<eos>"
    server = _replay_server((raw, "FINISHED_EOS"), tokenizer=_QwenChatTokenizer())
    with TestClient(server.app) as client:
        message, reason, _ = _collect_chat(client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Question"}], "stream": stream,
        }), stream)
    assert message["content"] == "hiddenAnswer"
    assert reason == "stop"
    _assert_released(server, 1)
