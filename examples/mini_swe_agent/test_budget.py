# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify history selection without requiring a model server or model files."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, models, pre_tokenizers

from budget import BudgetedTextModel, ContextBudgetExceeded
from context import workspace_context


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(folder / "tokenizer.json"))
        (folder / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}assistant "
        }))
        self.model = BudgetedTextModel(tokenizer_path=folder, context_limit=100,
                                      model_name="openai/test", model_kwargs={"max_tokens": 20})

    def messages(self):
        result = [{"role": "system", "content": "Keep requirements"},
                  {"role": "user", "content": "Repair the code without editing tests"}]
        for index in range(8):
            result.extend([
                {"role": "assistant", "content": f"command {index}"},
                {"role": "user", "content": "observed " * 20, "extra": {"raw_output": "observed " * 20}},
            ])
        return result

    def test_discards_old_pairs_and_preserves_task_and_latest_evidence(self):
        messages = self.messages()
        original = copy.deepcopy(messages)
        prepared = self.model._prepare_messages_for_api(messages)
        self.assertEqual(messages, original)
        self.assertEqual(prepared[:2], original[:2])
        self.assertEqual(prepared[-2]["content"], "command 7")
        self.assertEqual(prepared[-1]["content"], original[-1]["content"])
        self.assertLessEqual(self.model.counter.count(prepared) + 20, 100)
        dropped = self.model.context_events[-1]["dropped_message_indices"]
        self.assertTrue(dropped)
        for index in range(2, len(messages), 2):
            self.assertEqual(index in dropped, index + 1 in dropped)

    def test_user_followup_is_not_dropped(self):
        messages = self.messages()
        followup = {"role": "user", "content": "Also preserve the public API",
                    "extra": {"interrupt_type": "UserNewTask"}}
        messages.insert(6, followup)
        prepared = self.model._prepare_messages_for_api(messages)
        self.assertIn(followup["content"], [m["content"] for m in prepared])

    def test_large_observation_is_marked_and_original_retained(self):
        messages = self.messages()[:4]
        messages[-1]["content"] = "large output " * 1000
        original = messages[-1]["content"]
        prepared = self.model._prepare_messages_for_api(messages)
        self.assertLessEqual(self.model.counter.count(prepared) + 20, 100)
        self.assertIn("Observation shortened", prepared[-1]["content"])
        self.assertEqual(messages[-1]["content"], original)

    def test_oversized_task_is_rejected_instead_of_silently_shortened(self):
        with self.assertRaises(ContextBudgetExceeded):
            self.model._prepare_messages_for_api([
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "important requirement " * 100},
            ])

    def test_workspace_evidence_survives_trimming_and_refreshes_after_edit(self):
        folder = Path(self.temporary.name)
        (folder / "requirements.md").write_text("Preserve decimals and exact matching.")
        (folder / "source.py").write_text("# header\nresult = 'before'\n")
        self.model.context_limit = 200
        self.model.workspace = folder
        self.model.context_files = ["requirements.md", "source.py"]
        prepared = self.model._prepare_messages_for_api(self.messages())
        self.assertIn("Preserve decimals", prepared[2]["content"])
        self.assertIn("2|result = 'before'", prepared[2]["content"])
        self.assertTrue(self.model.context_events[-1]["dropped_message_indices"])
        first_hash = self.model.context_events[-1]["context_file_hashes"]["source.py"]
        (folder / "source.py").write_text("# header\nresult = 'after'\n")
        second = self.model._prepare_messages_for_api(self.messages())
        self.assertIn("result = 'after'", second[2]["content"])
        self.assertNotIn("result = 'before'", second[2]["content"])
        self.assertNotEqual(first_hash, self.model.context_events[-1]["context_file_hashes"]["source.py"])
        self.assertLessEqual(self.model.counter.count(second) + 20, 200)

    def test_workspace_evidence_cannot_be_silently_discarded(self):
        folder = Path(self.temporary.name)
        (folder / "source.py").write_text("important = 1\n" * 100)
        self.model.workspace = folder
        self.model.context_files = ["source.py"]
        with self.assertRaises(ContextBudgetExceeded):
            self.model._prepare_messages_for_api(self.messages())

    def test_snapshot_keeps_strings_line_numbers_and_reports_missing_file(self):
        folder = Path(self.temporary.name)
        source = '# license\n\ntext = """\n# not a comment\n"""\n'
        (folder / "source.py").write_text(source)
        evidence, hashes = workspace_context(folder, ["source.py", "missing.py"])
        self.assertNotIn("# license", evidence)
        self.assertIn('3|text = """\n4|# not a comment\n5|"""', evidence)
        self.assertIn("File missing.py: missing", evidence)
        self.assertIsNone(hashes["missing.py"])
        self.assertEqual((folder / "source.py").read_text(), source)

    def test_snapshot_rejects_paths_outside_workspace(self):
        with self.assertRaises(ValueError):
            workspace_context(Path(self.temporary.name), ["../outside.py"])

    def test_large_current_evidence_reduces_output_before_discarding_source(self):
        folder = Path(self.temporary.name)
        source = "value = 1\n" * 120
        (folder / "source.py").write_text(source)
        self.model.workspace = folder
        self.model.context_files = ["source.py"]
        self.model.context_limit = 2000
        self.model.config.model_kwargs["max_tokens"] = 1500
        prepared = self.model._prepare_messages_for_api([
            {"role": "system", "content": "rules"}, {"role": "user", "content": "fix the source"},
        ])
        event = self.model.context_events[-1]
        self.assertIn("120|value = 1", prepared[2]["content"])
        self.assertLess(event["output_reserve"], 1500)
        self.assertGreaterEqual(event["output_reserve"], 1024)
        self.assertLessEqual(event["prompt_tokens"] + event["output_reserve"], 2000)
        with patch("smoke.CompleteTextModel._query") as query:
            self.model._query(prepared)
        self.assertEqual(query.call_args.kwargs["max_tokens"], event["output_reserve"])
        self.assertEqual(self.model.config.model_kwargs["max_tokens"], 1500)

    def test_truncated_thinking_fallback_persists_until_new_user_task(self):
        folder = Path(self.temporary.name)
        (folder / "source.py").write_text("value = 1\n")
        self.model.workspace = folder
        self.model.context_files = ["source.py"]
        self.model.context_limit = 4096
        self.model.config.model_kwargs.update({
            "max_tokens": 2560, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        })
        messages = [
            {"role": "system", "content": "rules"}, {"role": "user", "content": "fix the source"},
            {"role": "user", "content": "Generation was truncated; no action executed.",
             "extra": {"response": {"choices": [{"finish_reason": "length"}]}}},
        ]
        prepared = self.model._prepare_messages_for_api(messages)
        with patch("smoke.CompleteTextModel._query") as query:
            self.model._query(prepared)
        self.assertFalse(query.call_args.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(query.call_args.kwargs["max_tokens"], 1024)
        self.assertTrue(self.model.context_events[-1]["recovery_without_thinking"])
        messages.extend([
            {"role": "assistant", "content": "```bash\ncat source.py\n```",
             "extra": {"response": {"choices": [{"finish_reason": "stop"}]}}},
            {"role": "user", "content": "value = 1"},
        ])
        self.model._prepare_messages_for_api(messages)
        self.assertFalse(self.model.counter.enable_thinking)
        self.assertEqual(self.model.context_events[-1]["recovery_reason"], "truncated_thinking")
        messages.append({"role": "user", "content": "Now solve a new task",
                         "extra": {"interrupt_type": "UserNewTask"}})
        self.model._prepare_messages_for_api(messages)
        self.assertTrue(self.model.counter.enable_thinking)
        self.assertFalse(self.model.context_events[-1]["recovery_without_thinking"])
        self.assertTrue(self.model.config.model_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_stalled_tool_results_activate_fallback_without_generation_truncation(self):
        folder = Path(self.temporary.name)
        (folder / "source.py").write_text("value = 1\n")
        self.model.workspace = folder
        self.model.context_files = ["source.py"]
        self.model.context_limit = 4096
        self.model.config.model_kwargs.update({
            "max_tokens": 2560, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        })
        messages = [
            {"role": "system", "content": "rules"}, {"role": "user", "content": "fix the source"},
            {"role": "assistant", "content": "```bash\ntrue\n```",
             "extra": {"response": {"choices": [{"finish_reason": "stop"}]}}},
            {"role": "user", "content": "No files changed.",
             "extra": {"raw_output": "", "stalled": True}},
        ]
        self.model._prepare_messages_for_api(messages)
        event = self.model.context_events[-1]
        self.assertFalse(event["enable_thinking"])
        self.assertEqual(event["recovery_reason"], "no_progress")
        self.assertEqual(event["output_reserve"], 1024)

    def test_stalled_history_is_removed_but_failed_acceptance_is_retained(self):
        folder = Path(self.temporary.name)
        (folder / "source.py").write_text("value = 1\n")
        self.model.workspace = folder
        self.model.context_files = ["source.py"]
        self.model.context_limit = 400
        messages = self.messages()[:2] + [
            {"role": "assistant", "content": "submit"},
            {"role": "user", "content": "check failed", "extra": {
                "raw_output": "check failed", "verification": {
                    "passed": False, "reasons": ["check exit 1"],
                    "output": json.dumps({"independent": {"output": "AssertionError: edge case"}}),
                },
            }},
            {"role": "assistant", "content": "BAD_REPEAT_COMMAND"},
            {"role": "user", "content": "No change", "extra": {"raw_output": "", "stalled": True}},
        ]
        prepared = self.model._prepare_messages_for_api(messages)
        text = "\n".join(m["content"] for m in prepared)
        self.assertNotIn("BAD_REPEAT_COMMAND", text)
        self.assertIn("AssertionError: edge case", text)
        self.assertIn("value = 1", text)
        self.assertTrue(self.model.context_events[-1]["history_restarted"])
        # The feedback also survives later ordinary history trimming.
        messages.extend(self.messages()[2:])
        prepared = self.model._prepare_messages_for_api(messages)
        self.assertIn("AssertionError: edge case", "\n".join(m["content"] for m in prepared))
        self.assertNotIn("BAD_REPEAT_COMMAND", "\n".join(m["content"] for m in prepared))
        self.assertFalse(self.model.context_events[-1]["history_restarted"])
        messages.append({"role": "user", "content": "new task",
                         "extra": {"interrupt_type": "UserNewTask"}})
        self.model._prepare_messages_for_api(messages)
        self.assertFalse(self.model.context_events[-1]["verification_evidence_retained"])


if __name__ == "__main__":
    unittest.main()
