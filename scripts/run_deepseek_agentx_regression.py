# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a frozen CI-matched DeepSeek + AgentX regression using an existing Conda environment."""

import argparse
import json
from pathlib import Path
import sys

from deepseek_agentx.common import PROFILE, RegressionError, load_profile, now, read_json, write_json


def nonnegative_seconds(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer number of seconds") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("seconds must be non-negative")
    return number


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--profile", type=Path, default=PROFILE)
    sub = command.add_subparsers(dest="action", required=True)
    sub.add_parser("plan", help="Print and validate the workload without side effects")
    observe = sub.add_parser("identity", help="Collect identities in the selected interpreter")
    observe.add_argument("--run-dir", type=Path, required=True)
    observe.add_argument("--output", type=Path, required=True)
    observe.add_argument("--label", default="host")
    observe.add_argument("--with-imports", action="store_true")
    for name in ("assess", "cleanup"):
        action = sub.add_parser(name)
        action.add_argument("--run-dir", type=Path, required=True)
    full = sub.add_parser("full")
    full.add_argument("--work-root", type=Path, required=True)
    full.add_argument("--run-dir", type=Path, help="New run directory, mainly for CI cleanup tracking")
    full.add_argument("--conda-env", required=True, help="Existing Conda environment name or absolute prefix")
    full.add_argument("--model", type=Path, required=True)
    full.add_argument("--tokenizer", type=Path, help="Defaults to the model directory")
    full.add_argument("--cann-root", type=Path, required=True)
    full.add_argument("--port", type=int, default=8232)
    full.add_argument("--index-url", default="https://pypi.tuna.tsinghua.edu.cn/simple")
    full.add_argument(
        "--task-timeout", type=nonnegative_seconds, default=None,
        help="Override task-submit --timeout; 0 waits indefinitely in the queue",
    )
    full.add_argument(
        "--task-max-time", type=nonnegative_seconds, default=None,
        help="Override task-submit --max-time; positive values must cover the profile server budget, 0 removes the limit",
    )
    return command


def main():
    args = parser().parse_args()
    try:
        if args.action == "plan":
            print(json.dumps(load_profile(args.profile), indent=2))
            return 0
        if args.action == "identity":
            from deepseek_agentx.identity import capture
            manifest = read_json(args.run_dir / "manifest.json")
            manifest["output"] = str(args.output.resolve())
            capture(manifest, args.label, args.with_imports)
            return 0
        if args.action == "cleanup":
            from deepseek_agentx.engine import cleanup, render
            if (args.run_dir / "shutdown.json").is_file():
                previous = read_json(args.run_dir / "shutdown.json")
                if previous.get("verified") is True and (args.run_dir / "result.json").is_file():
                    print(json.dumps(previous, indent=2))
                    return 0
            manifest = read_json(args.run_dir / "manifest.json") if (args.run_dir / "manifest.json").exists() else {}
            result = cleanup(args.run_dir.resolve(), manifest.get("port"), 120)
            if not (args.run_dir / "result.json").is_file():
                from deepseek_agentx.assessment import assess
                if not (args.run_dir / "finish-reason.json").is_file():
                    write_json(args.run_dir / "finish-reason.json", {"outcome": "INCOMPLETE", "at": now(),
                               "detail": "Controller did not finalize; external cleanup was required"})
                render(args.run_dir.resolve(), manifest, assess(args.run_dir.resolve()))
            print(json.dumps(result, indent=2))
            return 0 if result["verified"] else 1
        if args.action == "assess":
            from deepseek_agentx.assessment import assess
            result = assess(args.run_dir.resolve())
            print(json.dumps(result, indent=2))
            return 0 if result["outcome"] == "PASS_STABILITY" else 1
        if not 1 <= args.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        from deepseek_agentx.engine import run
        return run(args, load_profile(args.profile))
    except (RegressionError, ValueError, OSError) as exc:
        print(f"{getattr(exc, 'outcome', 'BLOCKED_SETUP')}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
