# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Own a temporary model server and run selected sequential repair comparisons."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.request


def stop_process_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # Also reap workers that outlive the leader after a failed server shutdown.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--port", default=8901, type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--serving-bin", default="/data/zhaomin/.conda/envs/zm_pypto/bin/pypto-serving")
    parser.add_argument("--agent-python", default="/data/zhaomin/agent/.venvs/mini-swe-agent/bin/python")
    parser.add_argument("--model-path", default="/data/models/Qwen3-14B")
    parser.add_argument("--repeats", type=int, default=1, help="Fresh workspaces per variant")
    parser.add_argument("--variants", nargs="+",
                        choices=("baseline", "output512", "thinking", "thinking1536",
                                 "thinking2048", "thinking2560", "thinking2560_grounded", "requirements"),
                        default=["baseline", "output512", "thinking", "requirements"])
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if len(args.variants) != len(set(args.variants)):
        parser.error("--variants must not contain duplicates; use a fresh suite for repeats")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_file = root / "suite.json"
    if state_file.exists():
        parser.error("Use a new output directory for each suite")
    plan = [(variant, repeat + 1) for repeat in range(args.repeats) for variant in args.variants]
    state = {"status": "starting", "device": args.device, "port": args.port,
             "order": args.variants, "repeats": args.repeats, "runs": []}

    def save():
        temporary = root / "suite.json.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_file)

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    server = child = None
    save()
    try:
        # Fail if the port is owned by another service; never reuse or stop it.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        implementation = root / "implementation"
        shutil.copytree(Path(__file__).resolve().parent, implementation,
                        ignore=shutil.ignore_patterns("__pycache__"))
        env = dict(os.environ)
        env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        command = [args.serving_bin, "--model", args.model_path, "--served-model-name", "qwen3-agent",
                   "--backend", "npu", "--platform", "a2a3", "--device", str(args.device),
                   "--max-model-len", "4096", "--max-num-seqs", "16", "--host", "127.0.0.1",
                   "--port", str(args.port)]
        state["server_command"] = command
        with (root / "server.log").open("w") as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        deadline = time.monotonic() + 240
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during startup: {server.returncode}")
            try:
                with opener.open(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Server did not become ready within 240 seconds")
            time.sleep(2)
        for variant, repeat in plan:
            label = f"{variant}_repeat{repeat}" if args.repeats > 1 else variant
            state["status"] = "running"
            state["current_variant"] = label
            save()
            print("Starting", variant, flush=True)
            folder = root / label
            folder.mkdir()
            command = [args.agent_python, str(implementation / "experiment.py"), "--variant", variant,
                       "--api-base", f"http://127.0.0.1:{args.port}/v1", "--tokenizer-path", args.model_path,
                       "--output-dir", str(folder)]
            with (root / (label + ".log")).open("w") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            try:
                code = child.wait(timeout=750)
            finally:
                stop_process_group(child)
                child = None
            summaries = list(folder.glob("*/summary.json"))
            if len(summaries) != 1:
                raise RuntimeError(f"Missing experiment summary for {variant}, exit={code}")
            result = json.loads(summaries[0].read_text())
            result["repeat"] = repeat
            state["runs"].append(result)
            save()
            print(variant, "accepted=", result["accepted"], "responses=", result["responses"], flush=True)
            if result["responses"] == 0 or server.poll() is not None:
                raise RuntimeError("Inference unavailable; stopped the suite")
        state["status"] = "completed"
        state.pop("current_variant", None)
    except BaseException as error:
        state["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "error"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop_process_group(child)
        stop_process_group(server)
        state["server_stopped"] = server is None or server.poll() is not None
        save()
    print("Suite results:", state_file, flush=True)
    # Completing a diagnostic comparison does not require every candidate to pass.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
