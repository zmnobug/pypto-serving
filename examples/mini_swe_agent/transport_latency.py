# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare fixed, warm model requests through HTTP and the direct engine."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time
import urllib.request

import litellm

from direct_engine import DirectEngineTextModel, EngineHost
from experiment_suite import stop_process_group


MESSAGES = [{"role": "user", "content": "Reply with the single word READY."}]
MAX_OUTPUT_TOKENS = 8


def probe_requests(generate, *, warmups: int, samples: int) -> dict:
    calls = []
    for index in range(warmups + samples):
        started = time.monotonic()
        response = generate()
        seconds = time.monotonic() - started
        if index < warmups:
            continue
        content = response.choices[0].message.content or ""
        calls.append({
            "seconds": seconds,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
            "text_sha256": hashlib.sha256(content.encode()).hexdigest(),
        })
    times = [call["seconds"] for call in calls]
    return {"calls": calls, "mean_seconds": statistics.mean(times),
            "median_seconds": statistics.median(times)}


def run_direct(args) -> dict:
    from pypto_serving.cli.main import build_parser, build_serving_engine_config

    engine_args = [
        "--model", str(args.model_path), "--served-model-name", "qwen3-agent",
        "--platform", args.platform, "--device", str(args.device),
        "--max-model-len", str(args.context_limit),
    ]
    config = build_serving_engine_config(build_parser().parse_args(engine_args))
    started = time.monotonic()
    with EngineHost(config) as host:
        startup_seconds = time.monotonic() - started
        model = DirectEngineTextModel(
            engine=host.engine, loop=host.loop,
            tokenizer_path=args.model_path, context_limit=args.context_limit,
            model_name="openai/qwen3-agent", action_regex=r"```bash\s*\n(.*?)\n```",
            cost_tracking="ignore_errors",
            model_kwargs={
                "temperature": 0.0, "max_tokens": MAX_OUTPUT_TOKENS, "timeout": args.request_timeout,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        )
        measured = probe_requests(lambda: model._query(MESSAGES), warmups=args.warmups, samples=args.samples)
    return {"transport": "direct", "startup_seconds": startup_seconds, **measured}


def wait_for_server(process, port: int) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 600
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"Serving exited during startup: {process.returncode}")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Serving did not become ready within 600 seconds")
        time.sleep(1)


def run_http(args) -> dict:
    with socket.socket() as port_check:
        port_check.bind(("127.0.0.1", args.port))
    command = [
        str(args.serving_bin), "--model", str(args.model_path),
        "--served-model-name", "qwen3-agent", "--backend", "npu",
        "--platform", args.platform, "--device", str(args.device),
        "--max-model-len", str(args.context_limit), "--max-num-seqs", "16",
        "--host", "127.0.0.1", "--port", str(args.port),
    ]
    process = None
    started = time.monotonic()
    with (args.output_dir / "server.log").open("w") as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ), start_new_session=True)
            wait_for_server(process, args.port)
            startup_seconds = time.monotonic() - started

            def generate():
                return litellm.completion(
                    model="openai/qwen3-agent", api_base=f"http://127.0.0.1:{args.port}/v1",
                    api_key="local-placeholder", messages=MESSAGES,
                    max_tokens=MAX_OUTPUT_TOKENS, temperature=0.0, stream=False,
                    timeout=args.request_timeout, num_retries=0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

            measured = probe_requests(generate, warmups=args.warmups, samples=args.samples)
        finally:
            stop_process_group(process)
    return {"transport": "http", "startup_seconds": startup_seconds, **measured}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/data/models/Qwen3-14B"))
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--serving-bin", type=Path,
                        default=Path("/data/zhaomin/.conda/envs/zm_pypto/bin/pypto-serving"))
    parser.add_argument("--port", type=int, default=8923)
    parser.add_argument("--order", choices=("direct,http", "http,direct"), default="direct,http")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.samples <= 0 or args.request_timeout <= 0:
        parser.error("Warmups must be nonnegative; samples and timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"]))
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    result = {"device": args.device, "order": args.order.split(","),
              "context_limit": args.context_limit, "max_tokens": MAX_OUTPUT_TOKENS,
              "thinking": False, "warmups": args.warmups, "samples": args.samples, "modes": []}
    for mode in result["order"]:
        try:
            measured = run_direct(args) if mode == "direct" else run_http(args)
            result["modes"].append(measured)
            (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2))
            print(mode, "startup=", round(measured["startup_seconds"], 3),
                  "median_request=", round(measured["median_seconds"], 4), flush=True)
        except BaseException as error:
            result["error"] = f"{type(error).__name__}: {error}"
            (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2))
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
