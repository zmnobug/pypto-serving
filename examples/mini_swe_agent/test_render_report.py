# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check offline report generation, including older runs and embedded shell text."""

import json
from pathlib import Path
import re
import tempfile
import unittest

from render_report import render_report


class RenderReportTests(unittest.TestCase):
    def test_report_embeds_events_without_executable_html(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "summary.json").write_text(json.dumps({
                "result": {"exit_status": "Submitted"}, "accepted": True,
                "elapsed_seconds": 2, "model_seconds": 1.5,
                "command_seconds_including_gate": .4,
            }))
            (run / "request_events.json").write_text(json.dumps([{
                "index": 0, "request_id": "serving-req-1", "status": "success",
                "finish_reason": "stop", "duration_seconds": 1.5,
            }]))
            command = "printf '</script><script>alert(1)</script>'"
            (run / "action_events.json").write_text(json.dumps([{
                "index": 0, "command": command, "returncode": 0,
                "duration_seconds": .4, "changed_files": [],
            }]))
            output = render_report(run / "summary.json")
            html = output.read_text()
            payload = re.search(r'<script id="report-data" type="application/json">(.*?)</script>',
                                html, re.DOTALL)
            self.assertIsNotNone(payload)
            self.assertEqual(json.loads(payload.group(1))["actions"][0]["command"], command)
            self.assertNotIn("</script><script>alert(1)</script>", html)
            self.assertEqual(output, run / "observation_report.html")

    def test_legacy_run_without_event_files_still_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "summary.json").write_text(json.dumps({
                "result": {"exit_status": "LimitsExceeded"}, "accepted": False,
            }))
            output = render_report(run, run / "report.html")
            self.assertTrue(output.is_file())
            self.assertIn('"requests": []', output.read_text())
            self.assertIn('"actions": []', output.read_text())


if __name__ == "__main__":
    unittest.main()
