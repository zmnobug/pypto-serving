# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check failure classification against minimal old and new trajectory shapes."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze_runs import analyze_run, main, trajectory_actions, verification_signature


class AnalyzeRunsTests(unittest.TestCase):
    def test_legacy_batch_keeps_actions_without_saved_observations(self):
        messages = [
            {"role": "assistant", "extra": {"actions": [{"command": "cat README.md"},
                                                       {"command": "python -m unittest"}]}},
            {"role": "user", "content": "<returncode>0</returncode>",
             "extra": {"raw_output": "requirements", "changed_files": []}},
            {"role": "exit", "extra": {"exit_status": "Submitted"}},
        ]
        actions = trajectory_actions(messages)
        self.assertEqual(len(actions), 2)
        self.assertEqual((actions[0]["observed"], actions[0]["returncode"]), (True, 0))
        self.assertFalse(actions[1]["observed"])
        self.assertIsNone(actions[1]["changed_files"])

    def test_repeated_verification_ignores_temporary_paths(self):
        def verification(path):
            return {"passed": False, "reasons": ["check exit 1"], "output": json.dumps({
                "public": {"output": "test_case (Tests) ... ERROR\n" + path},
                "independent": {"output": "PASS empty\nTraceback " + path},
            })}

        self.assertEqual(verification_signature(verification("/tmp/one")),
                         verification_signature(verification("/tmp/two")))

    def test_explicit_json_check_is_compared_with_completion_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {"passed": False, "public": {"output": "test_case (Tests) ... ERROR\n"},
                      "independent": {"output": "PASS empty\n"}}
            check = {"passed": False, "output": json.dumps(report),
                     "reasons": ["Verification command failed with exit code 1."]}
            (root / "summary.json").write_text(json.dumps({
                "result": {"exit_status": "TimeExceeded"}, "accepted": False,
            }))
            (root / "trajectory.json").write_text(json.dumps({"messages": [
                {"role": "user", "extra": {"raw_output": json.dumps(report) + "\n[Tracked files]"}},
                {"role": "user", "extra": {"verification": check}},
            ]}))
            analysis = analyze_run(root)
        self.assertEqual(analysis["finding_counts"]["explicit_failed_check"], 1)
        self.assertEqual(analysis["finding_counts"]["repeated_failed_check"], 1)

    def test_analyzer_joins_new_events_and_flags_repeated_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = {"result": {"exit_status": "TimeExceeded"}, "accepted": False,
                       "elapsed_seconds": 10, "model_seconds": 8,
                       "command_seconds_including_gate": 1, "responses": 2}
            check = {"passed": False, "reasons": ["check exit 1"], "output": "{}",
                     "current_files": {"source.py": "hash"}}
            trajectory = {"messages": [{"role": "user", "extra": {"verification": check}},
                                       {"role": "user", "extra": {"verification": check}}],
                          "info": {"context_management": {"requests": [{"dropped_message_indices": [2]}]}}}
            (root / "summary.json").write_text(json.dumps(summary))
            (root / "trajectory.json").write_text(json.dumps(trajectory))
            (root / "request_events.json").write_text(json.dumps([
                {"status": "success", "finish_reason": "length"},
                {"status": "error", "error_type": "TimeoutError"},
            ]))
            (root / "action_events.json").write_text(json.dumps([
                {"index": 0, "command": "sed -i s/a/b/ source.py", "changed_files": [], "stalled": True},
            ]))
            report = analyze_run(root)
        self.assertEqual(report["finding_counts"]["repeated_failed_check"], 1)
        self.assertEqual(report["finding_counts"]["possible_no_op_edit"], 1)
        self.assertEqual(report["finding_counts"]["context_trimmed"], 1)
        self.assertEqual(report["finding_counts"]["output_truncated"], 1)
        self.assertEqual(report["finding_counts"]["request_error"], 1)
        self.assertEqual(report["time_seconds"]["other_within_task"], 1)

    def test_cli_skips_non_agent_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "capacity").mkdir()
            (root / "capacity/summary.json").write_text("[]")
            (root / "repair").mkdir()
            (root / "repair/summary.json").write_text(json.dumps({
                "result": {"exit_status": "LimitsExceeded"}, "accepted": False,
            }))
            output = root / "analysis.json"
            with patch("sys.argv", ["analyze_runs.py", str(root), "--output", str(output)]):
                self.assertEqual(main(), 0)
            self.assertEqual(len(json.loads(output.read_text())), 1)


if __name__ == "__main__":
    unittest.main()
