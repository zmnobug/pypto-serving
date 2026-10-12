# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read the same compact Serving metrics from HTTP or an in-process engine."""

import json
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, build_opener


GAUGES = ("running", "waiting", "kv_cache_usage")
COUNTERS = (
    "prompt_tokens", "processed_prompt_tokens", "prefill_tokens", "generation_tokens",
    "prefix_cache_queries", "prefix_cache_hits", "preemptions", "requests_finished",
    "requests_error", "requests_aborted",
)


def compact_snapshot(snapshot: dict) -> dict:
    """Retain useful scheduler signals without copying large histogram arrays."""
    return {
        "server_id": snapshot.get("server_id"),
        "timestamp": snapshot.get("timestamp"),
        "replicas": [
            {
                "engine": replica.get("engine"),
                "gauges": {key: replica.get("gauges", {}).get(key) for key in GAUGES},
                "counters": {key: replica.get("counters", {}).get(key) for key in COUNTERS},
            }
            for replica in snapshot.get("replicas", [])
        ],
    }


def http_snapshot_reader(api_base: str, *, timeout: float = 0.5):
    """Read /metrics/json from the same origin as the Chat API, without proxying localhost."""
    parts = urlsplit(api_base)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("--api-base must be an HTTP(S) URL")
    url = urlunsplit((parts.scheme, parts.netloc, "/metrics/json", "", ""))
    opener = build_opener(ProxyHandler({}))

    def read():
        with opener.open(url, timeout=timeout) as response:
            return compact_snapshot(json.load(response))

    return read


def direct_snapshot_reader(engine):
    return lambda: compact_snapshot(engine.metrics.snapshot())
