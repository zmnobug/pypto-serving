# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check short and long-context HTTP inference before running Agent experiments."""

import json
from pathlib import Path
import re
import time
import urllib.request

from budget import LocalTextCounter


def probe_capacity(*, api_base: str, tokenizer_path: Path, context_limit: int,
                   output_dir: Path, timeout: float) -> list[dict]:
    if context_limit < 8192:
        raise ValueError("The long-context probe requires at least 8192 tokens")
    output_dir.mkdir(parents=True, exist_ok=False)
    counter = LocalTextCounter(tokenizer_path, enable_thinking=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = []
    for name, target in (("short", 0), ("cross_4096", 4090), ("long", context_limit - 1024)):
        code = "731942"

        def messages_for(count):
            return [{"role": "user", "content": (
                "Remember this exact code: " + code + ".\n"
                + "This line is irrelevant padding.\n" * count
                + "\nWhat was the exact code at the beginning? Return only those six digits."
            )}]

        low, high = 0, target
        while low < high:
            mid = (low + high) // 2
            if counter.count(messages_for(mid)) < target:
                low = mid + 1
            else:
                high = mid
        messages = messages_for(low)
        prompt_tokens = counter.count(messages)
        assert prompt_tokens + 512 <= context_limit
        body = {"model": "qwen3-agent", "messages": messages, "max_tokens": 512,
                "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": True}}
        (output_dir / f"{name}_request.json").write_text(json.dumps(body, indent=2))
        request = urllib.request.Request(api_base + "/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        started = time.monotonic()
        with opener.open(request, timeout=timeout) as response:
            data = json.loads(response.read())
        (output_dir / f"{name}_response.json").write_text(json.dumps(data, indent=2))
        choice = data["choices"][0]
        answer = re.sub(r"<think>.*?</think>", "", choice["message"].get("content") or "", flags=re.S).strip()
        result = {"name": name, "prompt_tokens": prompt_tokens, "usage": data["usage"],
                  "answer": answer, "finish_reason": choice["finish_reason"],
                  "enable_thinking": True, "seconds": time.monotonic() - started}
        result["passed"] = (answer == code and choice["finish_reason"] == "stop"
                            and data["usage"]["prompt_tokens"] == prompt_tokens
                            and (name == "short" or data["usage"]["total_tokens"] > 4096))
        results.append(result)
        (output_dir / "summary.json").write_text(json.dumps(results, indent=2))
        print("Capacity probe:", result, flush=True)
        if not result["passed"]:
            raise RuntimeError(f"Capacity probe failed: {name}; see {output_dir}")
    return results
