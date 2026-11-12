# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Write optional request and DSpark token diagnostics to per-process JSONL files."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def token_trace_enabled() -> bool:
    """Enable local diagnostic output only when an explicit directory is set."""
    return bool(os.environ.get("PYPTO_TOKEN_TRACE_DIR"))


def trace_tokens(event: str, request_id: str, **fields) -> None:
    """Append token diagnostics to a process-local file; never emit prompt text to logs."""
    directory = os.environ.get("PYPTO_TOKEN_TRACE_DIR")
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    record = {"time_ns": time.time_ns(), "event": event, "request_id": request_id, **fields}
    with (path / f"tokens-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
