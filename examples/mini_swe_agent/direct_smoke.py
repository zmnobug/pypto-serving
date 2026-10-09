# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the Agent-to-Serving-engine path without starting an HTTP server."""

import argparse
import json
from pathlib import Path
import secrets
import tempfile
import time

from minisweagent.agents.default import DefaultAgent

from direct_engine import DirectEngineTextModel, EngineHost
from smoke import SmokeEnvironment, SYSTEM_TEMPLATE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("/data/models/Qwen3-14B"))
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--use-compile-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mini_swe_agent_direct"))
    args = parser.parse_args()
    if args.context_limit <= 256:
        parser.error("--context-limit must exceed the 256-token output allowance")

    from pypto_serving.cli.main import build_parser, build_serving_engine_config

    engine_args = [
        "--model", str(args.model_path), "--served-model-name", "qwen3-agent",
        "--platform", args.platform, "--device", str(args.device),
        "--max-model-len", str(args.context_limit),
    ]
    if args.use_compile_cache:
        engine_args.append("--use-compile-cache")
    engine_config = build_serving_engine_config(build_parser().parse_args(engine_args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="roundtrip_", dir=args.output_dir.resolve()))
    workspace = run_dir / "workspace"
    workspace.mkdir()
    expected = f"hello_{secrets.token_hex(6)}"
    (workspace / "input.txt").write_text(expected + "\n")
    environment = SmokeEnvironment(cwd=str(workspace), timeout=10)
    started = time.monotonic()
    print("RUN_DIR", run_dir, flush=True)
    with EngineHost(engine_config) as host:
        model = DirectEngineTextModel(
            engine=host.engine, loop=host.loop,
            tokenizer_path=args.model_path, context_limit=args.context_limit,
            model_name="openai/qwen3-agent", action_regex=r"```bash\s*\n(.*?)\n```",
            cost_tracking="ignore_errors",
            model_kwargs={"temperature": 0.0, "max_tokens": 256, "timeout": 120,
                          "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        )
        agent = DefaultAgent(
            model, environment, system_template=SYSTEM_TEMPLATE, instance_template="{{task}}",
            step_limit=4, cost_limit=0, wall_time_limit_seconds=300,
            max_consecutive_format_errors=2, output_path=run_dir / "trajectory.json",
        )
        result = agent.run("First run exactly cat input.txt. Then submit its exact single-line content.")

    responses = [message["extra"]["response"] for message in agent.messages
                 if "response" in message.get("extra", {})]
    observations = [message.get("extra", {}).get("raw_output") for message in agent.messages]
    passed = (result.get("exit_status") == "Submitted"
              and result.get("submission", "").strip() == expected
              and expected + "\n" in observations
              and environment.commands[:1] == ["cat input.txt"]
              and agent.n_calls >= 2)
    summary = {"passed": passed, "transport": "direct-engine", "model_calls": agent.n_calls,
               "commands": environment.commands, "elapsed_seconds": round(time.monotonic() - started, 3),
               "usage": {key: sum((response.get("usage") or {}).get(key, 0) or 0 for response in responses)
                         for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
               "result": result}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
