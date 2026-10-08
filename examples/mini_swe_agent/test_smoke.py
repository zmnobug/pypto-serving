# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU checks for the integration's action execution boundaries."""

import tempfile
import unittest
from pathlib import Path

from litellm import ModelResponse
from minisweagent.exceptions import FormatError

from smoke import CompleteTextModel, SmokeEnvironment


class ActionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.model = CompleteTextModel(
            model_name="openai/qwen3-agent",
            cost_tracking="ignore_errors",
            action_regex=r"```bash\s*\n(.*?)\n```",
        )

    def response(self, content, reason):
        return ModelResponse(choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": reason,
        }])

    def test_success_reasons_accept_one_command(self):
        for reason in ("eos", "stop"):
            with self.subTest(reason=reason):
                parsed = self.model._parse_actions(self.response("```bash\ncat input.txt\n```", reason))
                self.assertEqual(parsed, [{"command": "cat input.txt"}])

    def test_incomplete_generation_rejects_even_complete_code_block(self):
        for reason in ("length", "aborted"):
            with self.subTest(reason=reason), self.assertRaises(FormatError):
                self.model._parse_actions(self.response("```bash\ncat input.txt\n```", reason))

    def test_missing_and_multiple_actions_rejected(self):
        for content in ("No action", "```bash\ncat input.txt\n```\n```bash\ncat input.txt\n```"):
            with self.subTest(content=content), self.assertRaises(FormatError):
                self.model._parse_actions(self.response(content, "stop"))

    def test_printed_action_wrapper_is_rejected_before_execution(self):
        for command in (
            "printf '%s\\n' 'RUN_SHELL_COMMAND' 'cat README.md'",
            'echo RUN_SHELL_COMMAND "cat README.md"',
        ):
            with self.subTest(command=command), self.assertRaises(FormatError) as caught:
                self.model._parse_actions(self.response(f"```bash\n{command}\n```", "stop"))
            self.assertIn("Nothing was executed", caught.exception.messages[0]["content"])
            self.assertEqual(caught.exception.messages[0]["extra"]["rejected_command"], command)

    def test_real_commands_and_completion_still_parse(self):
        for command in (
            "cat README.md",
            "rg RUN_SHELL_COMMAND README.md",
            "printf '%s\\n' 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' 'Done'",
        ):
            with self.subTest(command=command):
                self.assertEqual(
                    self.model._parse_actions(self.response(f"```bash\n{command}\n```", "stop")),
                    [{"command": command}],
                )

    def test_reasoning_commands_are_not_actions_and_original_is_preserved(self):
        content = "<think>```bash\necho discarded\n```</think>\n```bash\ncat input.txt\n```"
        response = self.response(content, "stop")
        self.assertEqual(self.model._parse_actions(response), [{"command": "cat input.txt"}])
        self.assertEqual(response.choices[0].message.content, content)
        with self.assertRaises(FormatError):
            self.model._parse_actions(self.response("<think>```bash\necho hidden\n```", "stop"))

    def test_multiple_final_actions_use_upstream_sequential_execution(self):
        self.model.allow_multiple_actions = True
        response = self.response("```bash\ncat a.txt\n```\n```bash\ncat b.txt\n```", "stop")
        self.assertEqual(self.model._parse_actions(response), [
            {"command": "cat a.txt"}, {"command": "cat b.txt"},
        ])
        response.choices[0].finish_reason = "length"
        with self.assertRaises(FormatError):
            self.model._parse_actions(response)

    def test_bad_later_action_rejects_entire_response(self):
        self.model.allow_multiple_actions = True
        for tail in ("echo RUN_SHELL_COMMAND 'cat a.txt'", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"):
            with self.subTest(tail=tail), self.assertRaises(FormatError):
                self.model._parse_actions(self.response(
                    "```bash\ncat a.txt\n```\n```bash\n" + tail + "\n```", "stop"))

    def test_batched_completion_is_last_and_requires_real_verification(self):
        from minisweagent.exceptions import Submitted
        from runtime import VerifiedEnvironment

        self.model.allow_multiple_actions = True
        self.model.verified_batch_completion = True
        content = "```bash\nprintf bad > result.txt\n```\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"
        actions = self.model._parse_actions(self.response(content, "stop"))
        with tempfile.TemporaryDirectory() as directory:
            env = VerifiedEnvironment(cwd=directory, require_changes=["result.txt"],
                                      verify_command="test \"$(cat result.txt)\" = good")
            outputs = [env.execute(action) for action in actions]
            self.assertEqual(outputs[-1]["returncode"], 1)
            self.assertFalse(env.verifications[-1]["passed"])
            actions[0]["command"] = "printf good > result.txt"
            with self.assertRaises(Submitted):
                for action in actions:
                    env.execute(action)
            self.assertTrue(env.verifications[-1]["passed"])
        with self.assertRaises(FormatError):
            self.model._parse_actions(self.response(
                "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```\n```bash\ntrue\n```", "stop"))

    def test_shell_chain_is_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "input.txt").write_text("probe\n")
            environment = SmokeEnvironment(cwd=directory)
            with self.assertRaises(ValueError):
                environment.execute({"command": "cat input.txt; touch unexpected"})
            self.assertFalse((workspace / "unexpected").exists())
            self.assertEqual(environment.commands, [])
            self.assertEqual(environment.execute({"command": "cat input.txt"})["output"], "probe\n")


if __name__ == "__main__":
    unittest.main()
