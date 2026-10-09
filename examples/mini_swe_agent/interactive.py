# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run upstream mini-swe-agent's interactive terminal against PyPTO Serving."""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile

import litellm
from minisweagent.agents.interactive import InteractiveAgent

from budget import BudgetedTextModel, ContextBudgetExceeded
from direct_engine import DirectGenerationError
from runtime import VerifiedEnvironment, coding_config, shell_environment


CHAT_SYSTEM_TEMPLATE = """You are a helpful assistant that can answer questions and use a shell.
Reply in the user's language. Be concise; summarize documents in at most six short
bullets. Every reply must contain exactly one bash code block. Its contents are
executed literally by a shell after user confirmation.

To read a file, execute the actual command directly, for example:
```bash
cat /path/to/README.md
```
To inspect part of a longer file:
```bash
sed -n '1,100p' /path/to/file
```
Wait for the returned file contents before summarizing. Never print a command
instead of executing it. RUN_SHELL_COMMAND is not a supported action or protocol.
A zero exit code only means the displayed shell command succeeded; printed command
text is not file content. Do not infer that a file is empty from unrelated output.
If reading fails, report the actual error instead of inventing document contents.

Only when you have enough evidence to answer, submit your answer with:
```bash
printf '%s\\n' 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' 'Your answer here'
```
Quote the answer as a shell literal. For greetings and general questions, use the
same completion command without inspecting files. Never answer in plain text,
even for a greeting. The following example shows the command format; adapt the
answer to the actual user message and translate it into the user's language.
For example, when the user says Hello, respond with:
```bash
printf '%s\\n' 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' 'Hello! How can I help?'
```
For file-specific questions, read the requested
file first. Commands run in separate shells in {{cwd}}; directory changes do not
persist. The user reviews commands before execution. Do not claim to have run a
command before seeing its result.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8899/v1")
    parser.add_argument("--transport", choices=("http", "direct"), default="http")
    parser.add_argument("--model", default="qwen3-agent")
    parser.add_argument("--model-path", type=Path, help="Model directory for direct mode; defaults to --tokenizer-path")
    parser.add_argument("--platform", default="a2a3", help="NPU platform in direct mode")
    parser.add_argument("--device", type=int, default=0, help="NPU device in direct mode")
    parser.add_argument("--use-compile-cache", action="store_true", help="Reuse compiled kernels in direct mode")
    parser.add_argument("--workspace", type=Path, help="Existing working directory for shell commands")
    parser.add_argument("--task", help="Initial message; otherwise prompt at startup")
    parser.add_argument("--profile", choices=("coding", "chat"), default="coding")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true", help="Keep Qwen thinking enabled on every request, including retries")
    parser.add_argument("--tokenizer-path", type=Path, help="Defaults to --model-path in direct mode")
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=600, help="Model request timeout in seconds")
    parser.add_argument("--verify-command", default="", help="Command that must pass before completion")
    parser.add_argument("--require-change", action="append", default=[], help="File that must change; repeatable")
    parser.add_argument("--protect", action="append", default=[], help="File that must stay unchanged; repeatable")
    parser.add_argument("--context-file", action="append", default=[],
                        help="Refresh and preserve this workspace file in every request; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mini_swe_agent_interactive"))
    args = parser.parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path or Path("/data/models/Qwen3-14B")
    if args.workspace is not None and not args.workspace.is_dir():
        parser.error("--workspace must be an existing directory")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.context_limit <= args.max_tokens:
        parser.error("--context-limit must exceed --max-tokens")
    if (args.require_change or args.protect) and not args.verify_command:
        parser.error("--require-change and --protect require --verify-command")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="session_", dir=args.output_dir.resolve()))
    workspace = args.workspace.resolve() if args.workspace else run_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
    os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1", "localhost"]))
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    print(f"Workspace: {workspace}")
    print(f"Trajectory: {run_dir / 'trajectory.json'}")
    print("At 'Execute': Enter approves the displayed command; a comment rejects it.")
    print("At 'Agent wants to finish': type your next message, or Enter to quit.")
    try:
        task = args.task or input("Your first message: ").strip()
    except (EOFError, KeyboardInterrupt):
        return 130
    if not task:
        return 0
    config = coding_config(compact=bool(args.context_file) and args.profile == "coding")
    environment = VerifiedEnvironment(
        cwd=str(workspace), timeout=30,
        env=shell_environment(config["environment"]["env"]),
        verify_command=args.verify_command, require_changes=args.require_change,
        protected_files=args.protect, verification_dir=str(run_dir),
        edit_feedback=bool(args.context_file),
    )
    system_template = config["agent"]["system_template"] if args.profile == "coding" else CHAT_SYSTEM_TEMPLATE
    if args.verify_command:
        system_template += (
            "\nCompletion is checked independently. Required changed files: " + str(args.require_change)
            + ". Protected files: " + str(args.protect)
            + ". Run this verification command exactly:\n" + args.verify_command
            + "\nFailed verification is returned as an observation; continue fixing the task.\n"
        )
    model_kwargs = {
        "temperature": 0.0, "max_tokens": args.max_tokens, "stream": False,
        "timeout": args.request_timeout, "num_retries": 0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": args.thinking}},
    }
    host = nullcontext(None)
    model_class = BudgetedTextModel
    if args.transport == "direct":
        from direct_engine import DirectEngineTextModel, EngineHost
        from pypto_serving.cli.main import build_parser, build_serving_engine_config

        engine_args = [
            "--model", str(args.model_path or args.tokenizer_path),
            "--served-model-name", args.model,
            "--platform", args.platform, "--device", str(args.device),
            "--max-model-len", str(args.context_limit),
        ]
        if args.use_compile_cache:
            engine_args.append("--use-compile-cache")
        host = EngineHost(build_serving_engine_config(build_parser().parse_args(engine_args)))
        model_class = DirectEngineTextModel
    else:
        model_kwargs.update({"api_base": args.api_base, "api_key": "local-placeholder"})

    with host as running_host:
        model = model_class(
            tokenizer_path=args.tokenizer_path, context_limit=args.context_limit,
            workspace=workspace, context_files=args.context_file,
            model_name=f"openai/{args.model}",
            action_regex=r"```bash\s*\n(.*?)\n```",
            allow_multiple_actions=True, verified_batch_completion=bool(args.verify_command),
            cost_tracking="ignore_errors",
            observation_template=config["model"]["observation_template"],
            format_error_template=config["model"]["format_error_template"],
            model_kwargs=model_kwargs,
            **({"engine": running_host.engine, "loop": running_host.loop} if running_host else {}),
        )
        agent = InteractiveAgent(
            model,
            environment,
            system_template=system_template,
            instance_template=config["agent"]["instance_template"] if args.profile == "coding" else "{{task}}",
            mode="confirm",
            confirm_exit=True,
            step_limit=20,
            cost_limit=0,
            max_consecutive_format_errors=2,
            output_path=run_dir / "trajectory.json",
        )
        try:
            result = agent.run(task)
        except EOFError:
            print(f"Session cancelled; task not accepted. Saved trajectory: {run_dir / 'trajectory.json'}")
            return 130
        except (litellm.APIConnectionError, litellm.Timeout, litellm.InternalServerError,
                litellm.ContextWindowExceededError, ContextBudgetExceeded,
                TimeoutError, DirectGenerationError) as error:
            print(f"Model request failed ({type(error).__name__}). Check model availability and context capacity.")
            print(f"The task is not complete. Saved trajectory: {run_dir / 'trajectory.json'}")
            return 1
    print(f"Session ended: {result.get('exit_status')}")
    return 0 if result.get("exit_status") == "Submitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
