# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the direct model boundary without loading weights or owning an NPU."""

import asyncio
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from minisweagent.exceptions import FormatError
from tokenizers import Tokenizer, models, pre_tokenizers

from direct_engine import DirectEngineTextModel, DirectGenerationError, EngineHost


class FakeTokenizer:
    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "rendered chat prompt"


class FakeEngine:
    def __init__(self, reason="FINISHED_EOS", text="```bash\ncat input.txt\n```"):
        self.tokenizer = FakeTokenizer()
        self.reason = reason
        self.text = text
        self.requests = []
        self.stopped = False

    def generate_request_id(self):
        return "direct-1"

    async def add_request(self, request_id, prompt, config):
        self.requests.append((request_id, prompt, config))
        yield SimpleNamespace(finished=True, finish_reason=self.reason, text=self.text,
                              prompt_tokens=19, completion_tokens=7)

    @staticmethod
    def normalize_finish_reason(reason):
        return {"FINISHED_EOS": "eos", "FINISHED_LENGTH": "length", "FINISHED_ABORTED": "aborted"}[reason]

    async def stop(self):
        self.stopped = True


class DirectEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(folder / "tokenizer.json"))
        (folder / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}assistant "
        }))
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()
        self.addCleanup(self.stop_loop)
        self.engine = FakeEngine()
        self.model = DirectEngineTextModel(
            engine=self.engine, loop=self.loop, generate_config_factory=SimpleNamespace,
            tokenizer_path=folder, context_limit=100, transport="direct",
            model_name="openai/qwen3-agent", action_regex=r"```bash\s*\n(.*?)\n```",
            cost_tracking="ignore_errors",
            model_kwargs={"max_tokens": 20, "temperature": 0.0, "timeout": 5,
                          "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        )

    def stop_loop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()

    def test_direct_query_preserves_prompt_budget_usage_and_action(self):
        self.model._request_max_tokens = 12
        response = self.model._query([
            {"role": "user", "content": "Read input.txt", "extra": {"local": True}},
        ])
        self.assertEqual(self.engine.tokenizer.messages, [{"role": "user", "content": "Read input.txt"}])
        self.assertEqual(self.engine.tokenizer.kwargs, {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": True,
        })
        request_id, prompt, config = self.engine.requests[0]
        self.assertEqual((request_id, prompt, config.max_new_tokens), ("direct-1", "rendered chat prompt", 12))
        self.assertFalse(config.stream)
        self.assertEqual(response.usage.prompt_tokens, 19)
        self.assertEqual(response.usage.completion_tokens, 7)
        self.assertEqual(response.id, "direct-1")
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertEqual(self.model.request_events[0]["request_id"], "direct-1")
        self.assertEqual(self.model.request_events[0]["transport"], "direct")
        self.assertEqual(self.model._parse_actions(response), [{"command": "cat input.txt"}])

    def test_length_response_is_recorded_and_never_executed(self):
        self.engine.reason = "FINISHED_LENGTH"
        with self.assertRaises(FormatError) as caught:
            self.model.query([{"role": "user", "content": "Read input.txt"}])
        self.assertEqual(caught.exception.messages[0]["extra"]["response"]["usage"]["completion_tokens"], 7)
        self.assertEqual(caught.exception.messages[0]["extra"]["finish_reason"], "length")

    def test_aborted_response_is_never_executed(self):
        self.engine.reason = "FINISHED_ABORTED"
        with self.assertRaises(FormatError) as caught:
            self.model.query([{"role": "user", "content": "Read input.txt"}])
        self.assertEqual(caught.exception.messages[0]["extra"]["finish_reason"], "aborted")

    def test_host_closes_engine_and_loop(self):
        host = EngineHost(config=SimpleNamespace())
        engine = FakeEngine()

        async def start_engine():
            return engine

        with patch.object(host, "_start_engine", start_engine):
            with host as started:
                self.assertIs(started.engine, engine)
                self.assertTrue(started.thread.is_alive())
        self.assertTrue(engine.stopped)
        self.assertFalse(host.thread.is_alive())

    def test_host_closes_loop_when_startup_fails(self):
        host = EngineHost(config=SimpleNamespace())

        async def fail_start():
            raise RuntimeError("worker unavailable")

        with patch.object(host, "_start_engine", fail_start), self.assertRaisesRegex(RuntimeError, "unavailable"):
            with host:
                pass
        self.assertFalse(host.thread.is_alive())
        self.assertTrue(host.loop.is_closed())

    def test_timeout_cancels_generation(self):
        cancelled = threading.Event()

        async def slow_request(_request_id, _prompt, _config):
            try:
                await asyncio.sleep(10)
                yield SimpleNamespace(finished=True, finish_reason="FINISHED_EOS", text="late",
                                      prompt_tokens=19, completion_tokens=1)
            finally:
                cancelled.set()

        self.engine.add_request = slow_request
        self.model.config.model_kwargs["timeout"] = 0.01
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            self.model._query([{"role": "user", "content": "Read input.txt"}])
        self.assertTrue(cancelled.wait(1))
        self.assertEqual(self.model.request_events[0]["error_type"], "TimeoutError")
        self.assertEqual(self.model.request_events[0]["request_id"], "direct-1")

    def test_engine_error_is_reported_as_generation_failure(self):
        async def failed_request(_request_id, _prompt, _config):
            raise ValueError("worker failed")
            yield

        self.engine.add_request = failed_request
        with self.assertRaises(DirectGenerationError) as caught:
            self.model._query([{"role": "user", "content": "Read input.txt"}])
        self.assertIsInstance(caught.exception.__cause__, ValueError)


if __name__ == "__main__":
    unittest.main()
