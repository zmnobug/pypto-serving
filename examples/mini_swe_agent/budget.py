# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fit text-action history to the local Qwen chat template and token budget."""

import json
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from context import workspace_context
from smoke import CompleteTextModel


class ContextBudgetExceeded(ValueError):
    """The pinned task and latest response cannot fit the configured context."""


class LocalTextCounter:
    """Count the local model's text-only chat template with its thinking setting."""

    def __init__(self, directory: Path, *, enable_thinking: bool = False):
        config = json.loads((directory / "tokenizer_config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.enable_thinking = enable_thinking
        self.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
            config["chat_template"]
        )

    def count(self, messages: list[dict]) -> int:
        # Match Serving's conversion: only role/content reach the chat template.
        text = self.template.render(
            messages=[{"role": m["role"], "content": m.get("content", "")} for m in messages],
            add_generation_prompt=True, enable_thinking=self.enable_thinking,
        )
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def shorten(self, content: str, keep: int) -> str:
        ids = self.tokenizer.encode(content, add_special_tokens=False).ids
        head = keep // 2
        tail = keep - head
        return (
            self.tokenizer.decode(ids[:head], skip_special_tokens=False)
            + "\n[Observation shortened to fit context. Full output is in the trajectory. "
              "Inspect a smaller file range if needed.]\n"
            + (self.tokenizer.decode(ids[-tail:], skip_special_tokens=False) if tail else "")
        )


class BudgetedTextModel(CompleteTextModel):
    def __init__(self, *, tokenizer_path: Path, context_limit: int,
                 workspace: Path | None = None, context_files: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        thinking = self.config.model_kwargs.get("extra_body", {}).get("chat_template_kwargs", {}).get(
            "enable_thinking", False
        )
        self.counter = LocalTextCounter(tokenizer_path, enable_thinking=thinking)
        self.context_limit = context_limit
        self.tokenizer_path = str(tokenizer_path)
        self.workspace = Path(workspace).resolve() if workspace is not None else None
        self.context_files = list(context_files or [])
        if self.context_files and self.workspace is None:
            raise ValueError("Context files require an explicit workspace")
        self.context_events: list[dict] = []
        self._request_max_tokens = self.config.model_kwargs["max_tokens"]
        self._task_marker = None
        self._history_restart_at = 0

    def _query(self, messages, **kwargs):
        overrides = {"max_tokens": self._request_max_tokens}
        # Preserve the user's thinking setting on every request, including retries.
        body = dict(self.config.model_kwargs.get("extra_body", {}))
        overrides["extra_body"] = body
        return super()._query(messages, **(kwargs | overrides))

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        max_tokens = self.config.model_kwargs["max_tokens"]
        thinking = self.config.model_kwargs.get("extra_body", {}).get("chat_template_kwargs", {}).get(
            "enable_thinking", False
        )
        task_start = max((i for i, m in enumerate(messages)
                          if i == 1 or m.get("extra", {}).get("interrupt_type") == "UserNewTask"), default=0)
        task_marker = (task_start, messages[task_start].get("content", "") if messages else "")
        if task_marker != self._task_marker:
            self._history_restart_at = 0
            self._task_marker = task_marker
        current_task = messages[task_start + 1:]
        # A reply can execute several actions. Inspect every observation since
        # that reply; a final failed submission must not hide an earlier stall.
        reply_index = max((i for i, m in enumerate(current_task)
                           if m.get("role") == "assistant" or "response" in m.get("extra", {})), default=-1)
        turn_stalled = any(m.get("extra", {}).get("stalled", False)
                           for m in current_task[reply_index + 1:])
        latest_response = next((m["extra"]["response"] for m in reversed(current_task)
                                if isinstance(m.get("extra", {}).get("response"), dict)), None)
        truncated = bool(latest_response and latest_response["choices"][0]["finish_reason"] == "length")
        self.counter.enable_thinking = thinking
        budget = self.context_limit - max_tokens
        evidence, evidence_hashes = (workspace_context(self.workspace, self.context_files)
                                     if self.context_files else ("", {}))
        pinned = set(range(min(2, len(messages))))
        groups = []
        for index, message in enumerate(messages[2:], 2):
            extra = message.get("extra", {})
            if extra.get("interrupt_type") in ("UserNewTask", "UserInterruption", "UserRejection"):
                pinned.add(index)
            elif message.get("role") == "assistant" or "response" in extra or not groups:
                groups.append([index])
            else:
                groups[-1].append(index)
        selected = set(range(len(messages)))
        shortened = {}
        restart_history = bool(self.context_files and (turn_stalled or truncated))
        if restart_history:
            # Do not keep presenting the same unsuccessful action as an example.
            self._history_restart_at = len(messages)
        if self._history_restart_at:
            selected = pinned | set(range(self._history_restart_at, len(messages)))
        verification = next((m["extra"]["verification"] for m in reversed(current_task)
                             if "verification" in m.get("extra", {})), None)

        def is_stale(check):
            checked_files = check.get("current_files", {})
            return any(name in checked_files and digest != checked_files[name]
                       for name, digest in evidence_hashes.items())

        stale_notice = (
            "This verification tested an older source snapshot. Files have changed since then; "
            "its failures do not describe the current code. Rerun verification before further edits. "
            "The original report remains in the trajectory."
        )
        stale_observations = {
            i: stale_notice for i, message in enumerate(messages)
            if self.context_files and "verification" in message.get("extra", {})
            and is_stale(message["extra"]["verification"])
        }
        verification_stale = bool(verification and is_stale(verification))
        verification_evidence = ""
        if self.context_files and verification_stale:
            verification_evidence = stale_notice + "\nVerification command: " + verification.get("command", "")
        elif self.context_files and verification and not verification.get("passed"):
            output = verification.get("output", "")
            try:
                report = json.loads(output)
            except (ValueError, TypeError):
                report = None
            if isinstance(report, dict):
                output = "\n".join(f"{name}:\n{item['output']}" for name, item in report.items()
                                   if isinstance(item, dict) and "output" in item)
            if len(self.counter.tokenizer.encode(output, add_special_tokens=False).ids) > 384:
                output = self.counter.shorten(output, 384)
            verification_evidence = (
                "Last completion check failed (historical evidence; rerun after edits):\n"
                + "\n".join(verification.get("reasons", [])) + "\n" + output
            )

        def prepare():
            candidates = [dict(messages[i], content=stale_observations.get(
                i, shortened.get(i, messages[i].get("content", ""))))
                          for i in sorted(selected)]
            if evidence:
                candidates.insert(2, {"role": "user", "content": evidence})
            if verification_evidence:
                candidates.append({"role": "user", "content": verification_evidence})
            if restart_history:
                restart_notice = (
                    "The previous generation reached its output limit; no command from it was executed. "
                    "Previous commands have been removed from this request. Thinking remains enabled. "
                    "Use the current source above and choose one next action. If you already edited the "
                    "source after a failed check, rerun verification instead of assuming the old error persists."
                ) if truncated else (
                    "Repeated actions made no progress. Previous commands have been removed from this request. "
                    "Start again from the current source and requirements above. Choose a different edit. "
                    "You may rewrite a small source file completely while preserving its public interface "
                    "and unrelated behavior. Do not modify tests or requirements."
                )
                candidates.append({"role": "user", "content": restart_notice})
            return super(BudgetedTextModel, self)._prepare_messages_for_api(candidates)

        prepared = prepare()
        prompt_tokens = self.counter.count(prepared)
        for group in groups[:-1]:
            if prompt_tokens <= budget:
                break
            selected.difference_update(group)
            prepared = prepare()
            prompt_tokens = self.counter.count(prepared)
        if self.context_files and not thinking and prompt_tokens > budget:
            # Non-thinking requests can trade output space for evidence. Keep
            # at least 1024 output tokens (or the smaller configured maximum).
            budget = self.context_limit - min(max_tokens, 1024)
        # Never shorten the task or user's follow-up instructions. If a single
        # observation is too large, preserve its beginning/end and mark the loss.
        for index in sorted(selected):
            if prompt_tokens <= budget:
                break
            if index in pinned or "raw_output" not in messages[index].get("extra", {}):
                continue
            content = messages[index].get("content", "")
            low, high = 0, len(self.counter.tokenizer.encode(content, add_special_tokens=False).ids)
            best = None
            while low <= high:
                keep = (low + high) // 2
                shortened[index] = self.counter.shorten(content, keep)
                candidate = prepare()
                if self.counter.count(candidate) <= budget:
                    best = shortened[index]
                    low = keep + 1
                else:
                    high = keep - 1
            shortened[index] = best if best is not None else self.counter.shorten(content, 0)
            prepared = prepare()
            prompt_tokens = self.counter.count(prepared)
        if prompt_tokens > budget:
            raise ContextBudgetExceeded(
                f"Pinned task, workspace evidence and latest action need {prompt_tokens} input tokens; "
                f"budget is {budget}. "
                "Start a fresh session or reduce the task/output allowance."
            )
        self._request_max_tokens = min(max_tokens, self.context_limit - prompt_tokens)
        event = {
            "prompt_tokens": prompt_tokens,
            "output_reserve": self._request_max_tokens,
            "enable_thinking": self.counter.enable_thinking,
            "recovery_without_thinking": False,
            "history_restarted": restart_history,
            "history_restart_reason": "truncated_generation" if restart_history and truncated else
                                      "stalled" if restart_history else "",
            "verification_evidence_retained": bool(verification_evidence),
            "verification_stale": verification_stale,
            "stale_verification_indices": sorted(stale_observations),
            "dropped_message_indices": sorted(set(range(len(messages))) - selected),
            "shortened_message_indices": sorted(shortened),
            "messages_sent": prepared,
            "context_file_hashes": evidence_hashes,
        }
        self.context_events.append(event)
        if self._request_max_tokens < self.config.model_kwargs["max_tokens"]:
            print(f"Context: preserving current evidence; output allowance is {self._request_max_tokens} tokens.",
                  flush=True)
        if event["dropped_message_indices"] or shortened:
            print(f"Context: {prompt_tokens} input tokens; older observations omitted from this request.", flush=True)
        return prepared

    def serialize(self):
        data = super().serialize()
        data.setdefault("info", {})["context_management"] = {
            "tokenizer_path": self.tokenizer_path,
            "context_limit": self.context_limit,
            "context_files": self.context_files,
            "workspace": str(self.workspace) if self.workspace is not None else None,
            "requests": self.context_events,
        }
        return data
