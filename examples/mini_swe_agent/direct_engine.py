# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Call the Serving engine from mini-swe-agent without HTTP or LiteLLM inference."""

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
import threading

from litellm import ModelResponse

from budget import BudgetedTextModel


class DirectGenerationError(RuntimeError):
    """The engine did not complete a direct generation request."""


class EngineHost:
    """Keep the async engine alive while the interactive Agent runs synchronously."""

    def __init__(self, config):
        self.config = config
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, name="pypto-agent-engine")
        self.engine = None

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()

    async def _start_engine(self):
        from pypto_serving.model.tokenizer import load_tokenizer
        from pypto_serving.serving.engine.async_engine import AsyncLLMEngine

        engine = AsyncLLMEngine(config=self.config, tokenizer=load_tokenizer(self.config.model_dir))
        await engine.start()
        return engine

    def __enter__(self):
        self.thread.start()
        try:
            self.engine = asyncio.run_coroutine_threadsafe(self._start_engine(), self.loop).result()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def close(self):
        try:
            if self.engine is not None and self.thread.is_alive():
                asyncio.run_coroutine_threadsafe(self.engine.stop(), self.loop).result()
                self.engine = None
        finally:
            if self.thread.is_alive():
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join()
            elif not self.loop.is_closed():
                self.loop.close()


class DirectEngineTextModel(BudgetedTextModel):
    """Preserve the existing Agent parser and budget while replacing inference."""

    def __init__(self, *, engine, loop, generate_config_factory=None, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.loop = loop
        self.generate_config_factory = generate_config_factory

    def _calculate_cost(self, _response):
        return {"cost": 0.0}

    async def _generate(self, messages, *, max_tokens, temperature, thinking):
        config_factory = self.generate_config_factory
        if config_factory is None:
            from pypto_serving.config.types import GenerateConfig
            config_factory = GenerateConfig

        prompt = self.engine.tokenizer.apply_chat_template(
            [{"role": message["role"], "content": message.get("content", "")} for message in messages],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
        )
        request_id = self.engine.generate_request_id()
        config = config_factory(max_new_tokens=max_tokens, temperature=temperature, stream=False)
        final = None
        try:
            async for output in self.engine.add_request(request_id, prompt, config):
                if output.finished:
                    final = output
        except Exception as error:
            raise DirectGenerationError(f"Generation failed for request {request_id}") from error
        if final is None or final.finish_reason == "error":
            raise DirectGenerationError(f"Generation failed for request {request_id}")
        reason = self.engine.normalize_finish_reason(final.finish_reason)
        if reason == "eos":
            reason = "stop"
        return ModelResponse(
            model=self.config.model_name.removeprefix("openai/"),
            choices=[{"index": 0, "message": {"role": "assistant", "content": final.text},
                      "finish_reason": reason,
                      "provider_specific_fields": {"native_finish_reason": reason}}],
            usage={"prompt_tokens": final.prompt_tokens,
                   "completion_tokens": final.completion_tokens,
                   "total_tokens": final.prompt_tokens + final.completion_tokens},
        )

    def _query(self, messages, **kwargs):
        options = self.config.model_kwargs | kwargs
        thinking = options.get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking", False)
        future = asyncio.run_coroutine_threadsafe(self._generate(
            messages, max_tokens=self._request_max_tokens,
            temperature=options.get("temperature", 0.0), thinking=thinking,
        ), self.loop)
        try:
            return future.result(timeout=options.get("timeout"))
        except FutureTimeoutError as error:
            future.cancel()
            raise TimeoutError("Direct engine generation timed out") from error
        except BaseException:
            future.cancel()
            raise
