# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Refresh explicitly selected workspace evidence without inventing a summary."""

import hashlib
from pathlib import Path


def workspace_context(workspace: Path, names: list[str]) -> tuple[str, dict]:
    workspace = workspace.resolve()
    sections = [
        "Current workspace evidence (refreshed before this request, not a proposed patch). "
        "Use these requirements and current source instead of guessing missing history. "
        "Numbers before | are original file line numbers, not part of the source."
    ]
    hashes = {}
    for name in dict.fromkeys(names):
        path = (workspace / name).resolve()
        if not path.is_relative_to(workspace):
            raise ValueError(f"Context path must stay inside the workspace: {name}")
        if not path.exists():
            sections.append(f"File {name}: missing.")
            hashes[name] = None
            continue
        if path.stat().st_size > 65536:
            raise ValueError(f"Context file exceeds 64 KiB; select a smaller file: {name}")
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        if "\x00" in text:
            raise ValueError(f"Context file must be text: {name}")
        hashes[name] = hashlib.sha256(raw).hexdigest()
        lines = text.splitlines()
        start = 0
        # Only omit the leading comment/blank header, never comments inside code
        # or strings. Preserve original line numbers and all executable source.
        if path.suffix == ".py":
            while start < len(lines) and (not lines[start].strip() or lines[start].lstrip().startswith("#")):
                start += 1
        header = f"File {name}: {len(lines)} lines"
        if start:
            header += f" (leading comment/blank header, lines 1-{start}, omitted)"
        body = "\n".join(f"{i + 1}|{line}" for i, line in enumerate(lines[start:], start))
        sections.append(header + "\n" + (body or "[empty source]"))
    return "\n\n".join(sections), hashes
