# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a bounded, unattended repair experiment on a fresh fixture copy."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from minisweagent.agents.default import DefaultAgent

from budget import BudgetedTextModel
from repair_demo import FIXTURE, run_tests
from runtime import VerifiedEnvironment, coding_config, shell_environment


TASK = (
    "Repair this serving log report project. Read README.md for the requirements, "
    "run python -m unittest -v to reproduce failures, inspect the source, fix it, "
    "then rerun tests and briefly explain the result. Only modify report.py and cli.py. "
    "Do not change tests or README and do not install dependencies. Keep replies concise."
)
VARIANTS = {
    "baseline": {"max_tokens": 1024, "thinking": False, "include_readme": False},
    "output512": {"max_tokens": 512, "thinking": False, "include_readme": False},
    "thinking": {"max_tokens": 1024, "thinking": True, "include_readme": False},
    "thinking1536": {"max_tokens": 1536, "thinking": True, "include_readme": False},
    "thinking2048": {"max_tokens": 2048, "thinking": True, "include_readme": False},
    "thinking2560": {"max_tokens": 2560, "thinking": True, "include_readme": False},
    "thinking2560_grounded": {"max_tokens": 2560, "thinking": True, "include_readme": False,
                              "context_files": ["README.md", "report.py", "cli.py"]},
    "thinking8192": {"max_tokens": 8192, "thinking": True, "include_readme": False,
                     "context_files": ["README.md", "report.py", "cli.py"]},
    "requirements": {"max_tokens": 1024, "thinking": False, "include_readme": True},
}


class TimedModel(BudgetedTextModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query_seconds = []

    def query(self, messages):
        started = time.monotonic()
        try:
            return super().query(messages)
        finally:
            self.query_seconds.append(time.monotonic() - started)


class LoggedEnvironment(VerifiedEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.command_seconds = []

    def execute(self, action, **kwargs):
        print("ACTION", action["command"], flush=True)
        started = time.monotonic()
        try:
            result = super().execute(action, **kwargs)
            print("RESULT", result["returncode"], result["output"][-1800:], flush=True)
            return result
        finally:
            self.command_seconds.append(time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8901/v1")
    parser.add_argument("--model", default="qwen3-agent")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("/data/models/Qwen3-14B"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mini_swe_agent_experiments"))
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--wall-time", type=float, default=600)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--prepare-only", action="store_true", help="Validate the initial request without inference")
    args = parser.parse_args()
    if args.context_limit <= VARIANTS[args.variant]["max_tokens"]:
        parser.error("--context-limit must exceed the configured output allowance")
    if args.request_timeout <= 0 or args.wall_time <= 0:
        parser.error("Timeouts must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=args.variant + "_", dir=args.output_dir.resolve()))
    workspace = run_dir / "workspace"
    shutil.copytree(FIXTURE, workspace, ignore=shutil.ignore_patterns("__pycache__"))
    variant = VARIANTS[args.variant]
    task = TASK
    if variant["include_readme"]:
        task += "\n\nOriginal README.md, supplied verbatim as task requirements:\n" + (workspace / "README.md").read_text()
    (run_dir / "task.txt").write_text(task)
    source_dir = Path(__file__).resolve().parent
    # Record the exact implementation and fixture used, outside the candidate workspace.
    snapshot = run_dir / "implementation"
    snapshot.mkdir()
    for path in source_dir.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    manifest = {
        "variant": args.variant, "settings": variant, "steps": args.steps,
        "context_limit": args.context_limit,
        "request_timeout": args.request_timeout, "wall_time": args.wall_time, "temperature": 0.0, "api_base": args.api_base,
        "model": args.model, "operator_hints": False, "agent": "DefaultAgent",
        "fixture_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in workspace.iterdir() if p.is_file()},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "baseline.json").write_text(json.dumps(run_tests(workspace), indent=2))
    os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
    os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"]))
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    context_files = variant.get("context_files", [])
    config = coding_config(compact=bool(context_files))
    verify_command = shlex.join([sys.executable, str(source_dir / "repair_demo.py"), "check", "."])
    required, protected = ["report.py", "cli.py"], ["README.md", "test_report.py"]
    environment = LoggedEnvironment(
        cwd=str(workspace), timeout=30, env=shell_environment(config["environment"]["env"]),
        verify_command=verify_command, require_changes=required, protected_files=protected,
        verification_dir=str(run_dir),
        edit_feedback=bool(context_files),
    )
    system_template = config["agent"]["system_template"] + (
        "\nCompletion is checked independently. Required changed files: " + str(required)
        + ". Protected files: " + str(protected)
        + ". Run this verification command exactly:\n" + verify_command
        + "\nFailed verification is returned as an observation; continue fixing the task.\n"
    )
    model = TimedModel(
        tokenizer_path=args.tokenizer_path, context_limit=args.context_limit, model_name=f"openai/{args.model}",
        workspace=workspace, context_files=context_files,
        action_regex=r"```bash\s*\n(.*?)\n```", allow_multiple_actions=True, verified_batch_completion=True, cost_tracking="ignore_errors",
        observation_template=config["model"]["observation_template"],
        format_error_template=config["model"]["format_error_template"],
        model_kwargs={"api_base": args.api_base, "api_key": "local-placeholder", "temperature": 0.0,
                      "max_tokens": variant["max_tokens"], "stream": False, "timeout": args.request_timeout,
                      "num_retries": 0,
                      "extra_body": {"chat_template_kwargs": {"enable_thinking": variant["thinking"]}}},
    )
    agent = DefaultAgent(
        model, environment, system_template=system_template,
        instance_template=config["agent"]["instance_template"], step_limit=args.steps,
        wall_time_limit_seconds=args.wall_time, cost_limit=0, max_consecutive_format_errors=2,
        output_path=run_dir / "trajectory.json",
    )
    print("RUN_DIR", run_dir, flush=True)
    if args.prepare_only:
        agent.extra_template_vars["task"] = task
        messages = [
            {"role": "system", "content": agent._render_template(system_template)},
            {"role": "user", "content": agent._render_template(config["agent"]["instance_template"])},
        ]
        model._prepare_messages_for_api(messages)
        (run_dir / "dry_run.json").write_text(json.dumps(model.context_events[-1], indent=2))
        print("Prepared only; no model request or Agent action executed.", flush=True)
        return 0
    started = time.monotonic()
    try:
        result = agent.run(task)
    except Exception as error:
        result = {"exit_status": type(error).__name__, "error": str(error)}
    elapsed = time.monotonic() - started
    try:
        check = subprocess.run([sys.executable, str(source_dir / "repair_demo.py"), "check", str(workspace)],
                               text=True, capture_output=True, timeout=90)
        (run_dir / "final_check.log").write_text(check.stdout + check.stderr)
        verification = json.loads(check.stdout) if check.returncode in (0, 1) else {}
    except (subprocess.TimeoutExpired, OSError, ValueError) as error:
        verification = {"passed": False, "error": str(error)}
    responses = [m["extra"]["response"] for m in agent.messages if "response" in m.get("extra", {})]
    events = model.context_events
    public = verification.get("public", {}).get("output", "")
    extra = verification.get("independent", {}).get("output", "")
    summary = {
        **manifest, "run_dir": str(run_dir), "result": result,
        "passed": verification.get("passed", False),
        "accepted": result.get("exit_status") == "Submitted" and verification.get("passed", False),
        "elapsed_seconds": elapsed, "model_seconds": sum(model.query_seconds),
        "command_seconds_including_gate": sum(environment.command_seconds),
        "calls_attempted": agent.n_calls, "responses": len(responses),
        "input_tokens": sum(r["usage"]["prompt_tokens"] for r in responses),
        "output_tokens": sum(r["usage"]["completion_tokens"] for r in responses),
        "max_prompt_tokens": max((r["usage"]["prompt_tokens"] for r in responses), default=0),
        "length_responses": sum(r["choices"][0]["finish_reason"] == "length" for r in responses),
        "trimmed_requests": sum(bool(e["dropped_message_indices"] or e["shortened_message_indices"]) for e in events),
        "recovery_requests": sum(e.get("recovery_without_thinking", False) for e in events),
        "all_requests_thinking": bool(events) and all(e["enable_thinking"] for e in events),
        "output_reserves": [e["output_reserve"] for e in events],
        "counts_match": len(events) == len(responses) and all(
            e["prompt_tokens"] == r["usage"]["prompt_tokens"] for e, r in zip(events, responses)),
        "public_passed": sum(line.startswith("test_") and line.endswith(" ... ok") for line in public.splitlines()),
        "independent_passed_before_stop": sum(line.startswith("PASS ") for line in extra.splitlines()),
        "verification_attempts": len(environment.verifications),
        "verification_rejections": sum(not v["passed"] for v in environment.verifications),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
