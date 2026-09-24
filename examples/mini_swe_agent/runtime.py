# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Use upstream coding prompts and verify completion against local evidence."""

import ast
import hashlib
import json
import os
from pathlib import Path
import sys

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.exceptions import Submitted


def coding_config(*, compact: bool = False) -> dict:
    config = get_config_from_spec(builtin_config_dir / "default.yaml")
    for section, key in (("agent", "system_template"), ("agent", "instance_template"),
                         ("model", "format_error_template")):
        config[section][key] = config[section][key].replace("mswea_bash_command", "bash")
    config["agent"]["system_template"] += (
        "\nReply in the user's language. Explain the next action briefly. "
        "Commands in bash blocks execute literally; do not print RUN_SHELL_COMMAND wrappers. "
        "A successful shell exit does not prove a replacement changed a file. "
        "After editing, inspect the result and run the tests before claiming success. "
        "If an edit fails, inspect the affected lines instead of repeating the same command. "
        "Read the project's requirements before editing; do not guess behavior from one assertion. "
        "For Python source edits, prefer literal replacements with an assertion over sed regular "
        "expressions (brackets and asterisks in code change regex meaning). For example:\n"
        "```bash\npython - <<'PY'\nfrom pathlib import Path\np = Path('module.py')\n"
        "text = p.read_text()\nold = 'return old_value'\nnew = 'return new_value'\n"
        "assert text.count(old) == 1, 'Expected exactly one target; inspect the file again'\n"
        "p.write_text(text.replace(old, new))\nPY\n```\n"
    )
    if compact:
        config["agent"]["system_template"] = (
            "You repair code using a local shell. Reply in the user's language. "
            "Produce exactly one bash code block with a real command per turn. "
            "Keep reasoning brief and focus on the next action, not a plan for every possible case. "
            "Read the supplied requirements and current source. Run the tests to reproduce, edit, "
            "then rerun them. Do not guess missing code, functions or requirements. "
            "Current workspace snapshots are refreshed each turn; use them even if old history is removed. "
            "Each command starts a new shell in {{cwd}}. "
            "For edits, you may rewrite a small file, insert at a known position, or use literal Python "
            "replacements with an assertion that the target occurs once. "
            "Do not use sed -i for source edits: code brackets are regex operators. For example:\n"
            "```bash\npython - <<'PY'\nfrom pathlib import Path\np = Path('module.py')\n"
            "s = p.read_text()\nold = 'return old_value'\nnew = 'return new_value'\n"
            "assert s.count(old) == 1\np.write_text(s.replace(old, new))\nPY\n```\n"
            "Replace the original expression; do not append a competing definition or duplicate dict key. "
            "After editing inspect the changed code. Empty output or unchanged files is not progress; "
            "check actual line counts instead of requesting ever higher empty ranges. "
            "Only finish after validation, with one bash block containing "
            "printf '%s\\n' 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' 'Brief result'. "
            "Never print RUN_SHELL_COMMAND wrappers.\n"
        )
        config["agent"]["instance_template"] = "{{task}}"
    return config


def shell_environment(extra: dict | None = None) -> dict:
    return dict(extra or {}) | {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "VIRTUAL_ENV": sys.prefix,
        "PYTHONNOUSERSITE": "1",
    }


class VerifiedEnvironmentConfig(LocalEnvironmentConfig):
    verify_command: str = ""
    require_changes: list[str] = []
    protected_files: list[str] = []
    verification_dir: str = ""
    verification_timeout: int = 60
    edit_feedback: bool = False


class VerifiedEnvironment(LocalEnvironment):
    """Reject completion until configured file checks and tests actually pass."""

    def __init__(self, **kwargs):
        super().__init__(config_class=VerifiedEnvironmentConfig, **kwargs)
        if (self.config.require_changes or self.config.protected_files) and not self.config.verify_command:
            raise ValueError("File acceptance checks require a verification command")
        self.workspace = Path(self.config.cwd).resolve()
        self.paths = {}
        for name in self.config.require_changes + self.config.protected_files:
            path = (self.workspace / name).resolve()
            if not path.is_relative_to(self.workspace):
                raise ValueError(f"Acceptance path must stay inside the workspace: {name}")
            self.paths[name] = path
        self.initial = self._snapshot()
        self.verifications: list[dict] = []
        self._last_attempt = None
        self._same_attempts = 0
        self._empty_observations = 0

    def _snapshot(self) -> dict:
        return {name: hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                for name, path in self.paths.items()}

    def execute(self, action: dict, **kwargs) -> dict:
        before = self._snapshot()
        try:
            output = super().execute(action, **kwargs)
        except Submitted as completed:
            if not self.config.verify_command:
                raise
            verification = self.verify()
            if verification["passed"]:
                completed.messages[0]["extra"]["verification"] = verification
                print("Completion verified: required file changes and checks passed.", flush=True)
                raise
            print("Completion rejected: verification failed; returning evidence to the Agent.", flush=True)
            return {
                "returncode": 1,
                "output": "Completion rejected. Fix the remaining issues and rerun tests.\n"
                          + "\n".join(verification["reasons"]) + "\n" + verification["output"],
                "exception_info": "",
                "extra": {"verification": verification},
            }
        after = self._snapshot()
        if self.config.edit_feedback:
            raw_output = output["output"]
            signature = (action.get("command", ""), output["returncode"], raw_output, after)
            self._same_attempts = self._same_attempts + 1 if signature == self._last_attempt else 1
            self._last_attempt = signature
            self._empty_observations = (
                self._empty_observations + 1 if before == after and not raw_output.strip() else 0
            )
            if self._same_attempts >= 3 or self._empty_observations >= 2:
                output.setdefault("extra", {})["stalled"] = True
                sizes = ", ".join(
                    f"{name}: {len(path.read_text().splitlines())} lines"
                    for name, path in self.paths.items() if path.is_file()
                )
                output["output"] += (
                    "\n[No progress: repeated identical results or empty output without file changes. "
                    "Inspect the current source and choose a different action; do not assume missing lines exist. "
                    "Replacing an absent substring cannot insert text: use an existing anchor or explicit insertion. "
                    + sizes + "]\n"
                )
            for name, path in self.paths.items():
                if before[name] == after[name] or path.suffix != ".py" or not path.is_file():
                    continue
                try:
                    tree = ast.parse(path.read_text())
                except SyntaxError as error:
                    output["output"] += f"\n[Edit check: {name}:{error.lineno}: {error.msg}]\n"
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Dict):
                        continue
                    keys = set()
                    for key in node.keys:
                        try:
                            value = ast.literal_eval(key)
                            hash(value)
                        except (ValueError, TypeError):
                            continue
                        if value in keys:
                            output["output"] += (
                                f"\n[Edit check: {name}:{key.lineno}: duplicate dictionary key {value!r}; "
                                "the later value overrides the earlier one. Replace the original entry.]\n"
                            )
                        keys.add(value)
        if self.paths:
            changed = [name for name in self.paths if before[name] != after[name]]
            output["output"] += "\n[Tracked files changed by this command: " + (", ".join(changed) or "none") + "]\n"
            if not changed and "sed -i" in action.get("command", ""):
                output["output"] += (
                    "The edit made no changes. sed can exit zero without matching anything. "
                    "Inspect the exact source and use a literal Python replacement with an assertion; "
                    "do not claim this edit succeeded or repeat it unchanged.\n"
                )
            output.setdefault("extra", {})["changed_files"] = changed
        return output

    def verify(self) -> dict:
        before = self._snapshot()
        checker = LocalEnvironment(cwd=str(self.workspace), env=self.config.env,
                                   timeout=self.config.verification_timeout)
        try:
            output = checker.execute({"command": self.config.verify_command})
        except Submitted:
            output = {"returncode": 1, "output": "Verification command emitted an Agent completion marker."}
        after = self._snapshot()
        reasons = []
        for name in self.config.require_changes:
            if after[name] == self.initial[name]:
                reasons.append(f"Required source file is unchanged: {name}")
        for name in self.config.protected_files:
            if after[name] != self.initial[name]:
                reasons.append(f"Protected file was modified: {name}")
        if before != after:
            reasons.append("Verification modified tracked files; recheck the candidate before submitting.")
        if output["returncode"] != 0:
            reasons.append(f"Verification command failed with exit code {output['returncode']}.")
        verification = {
            "passed": not reasons, "command": self.config.verify_command,
            "returncode": output["returncode"], "output": output["output"],
            "reasons": reasons, "initial_files": self.initial, "current_files": after,
        }
        self.verifications.append(verification)
        if self.config.verification_dir:
            folder = Path(self.config.verification_dir)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"verification_{len(self.verifications)}.json").write_text(json.dumps(verification, indent=2))
        return verification
