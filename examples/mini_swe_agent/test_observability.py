# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the shared Serving metrics shape without a live server."""

import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment import response_ids_match
from observability import compact_snapshot, direct_snapshot_reader, http_snapshot_reader


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.raw = {
            "server_id": "serving-1", "timestamp": 123.0,
            "replicas": [{"engine": 2, "gauges": {"running": 1, "waiting": 3, "kv_cache_usage": 0.5},
                          "counters": {"prompt_tokens": 20, "prefix_cache_hits": 8, "preemptions": 1},
                          "histograms": {"e2e": {"buckets": [1] * 1000}}}],
        }

    def test_direct_and_http_return_the_same_compact_schema(self):
        engine = SimpleNamespace(metrics=SimpleNamespace(snapshot=lambda: self.raw))
        direct = direct_snapshot_reader(engine)()
        self.assertEqual(direct, compact_snapshot(self.raw))
        self.assertNotIn("histograms", json.dumps(direct))

        class FakeOpener:
            def open(self, url, timeout):
                self_url.append(url)
                self_timeout.append(timeout)
                return io.BytesIO(json.dumps(self_raw).encode())

        self_url, self_timeout, self_raw = [], [], self.raw
        with patch("observability.build_opener", return_value=FakeOpener()):
            reader = http_snapshot_reader("http://127.0.0.1:8899/v1", timeout=0.5)
            self.assertEqual(reader(), direct)
        self.assertEqual(self_url, ["http://127.0.0.1:8899/metrics/json"])
        self.assertEqual(self_timeout, [0.5])

    def test_invalid_http_base_rejected(self):
        with self.assertRaises(ValueError):
            http_snapshot_reader("not-a-url")

    def test_request_response_join_ignores_failed_inference(self):
        events = [
            {"status": "error", "request_id": "failed"},
            {"status": "success", "request_id": "completed"},
        ]
        self.assertTrue(response_ids_match(events, [{"id": "completed"}]))
        self.assertFalse(response_ids_match(events, [{"id": "other"}]))
        self.assertFalse(response_ids_match(events, []))


if __name__ == "__main__":
    unittest.main()
