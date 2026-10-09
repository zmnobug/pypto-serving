# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check that transport timing excludes warmup requests."""

from types import SimpleNamespace
import unittest

from transport_latency import probe_requests


class TransportLatencyTests(unittest.TestCase):
    def test_warmup_is_excluded_but_all_requests_are_sent(self):
        calls = []

        def generate():
            calls.append(len(calls))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=str(len(calls))),
                                         finish_reason="length")],
                usage=SimpleNamespace(prompt_tokens=9, completion_tokens=1),
            )

        result = probe_requests(generate, warmups=2, samples=3)
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(result["calls"]), 3)
        self.assertTrue(all(call["prompt_tokens"] == 9 and call["completion_tokens"] == 1
                            for call in result["calls"]))
        self.assertGreaterEqual(result["median_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
