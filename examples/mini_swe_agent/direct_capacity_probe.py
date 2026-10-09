# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify direct generation across the Qwen 4K context boundary."""

import argparse
from pathlib import Path

from capacity_probe import probe_capacity
from direct_engine import DirectEngineTextModel, EngineHost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/data/models/Qwen3-14B"))
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--use-compile-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from pypto_serving.cli.main import build_parser, build_serving_engine_config

    engine_args = [
        "--model", str(args.model_path), "--served-model-name", "qwen3-agent",
        "--platform", args.platform, "--device", str(args.device),
        "--max-model-len", str(args.context_limit),
    ]
    if args.use_compile_cache:
        engine_args.append("--use-compile-cache")
    engine_config = build_serving_engine_config(build_parser().parse_args(engine_args))
    with EngineHost(engine_config) as host:
        model = DirectEngineTextModel(
            engine=host.engine, loop=host.loop,
            tokenizer_path=args.model_path, context_limit=args.context_limit,
            model_name="openai/qwen3-agent", action_regex=r"```bash\s*\n(.*?)\n```",
            cost_tracking="ignore_errors",
            model_kwargs={
                "temperature": 0.0, "max_tokens": 512, "timeout": args.request_timeout,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            },
        )
        results = probe_capacity(
            api_base=None, tokenizer_path=args.model_path, context_limit=args.context_limit,
            output_dir=args.output_dir, timeout=args.request_timeout,
            generate=lambda body: model._query(body["messages"]).model_dump(),
        )
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
