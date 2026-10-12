# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stable failure signatures shared by offline analysis and optional Agent feedback."""

import json
import re


FAILED_TEST = re.compile(r"^(test_\S+ .* \.\.\. (?:FAIL|ERROR))$", re.MULTILINE)


def verification_signature(verification: dict) -> dict:
    """Compare failed checks by failed tests, not unstable temporary paths."""
    raw_output = verification.get("output", "")
    try:
        report = json.loads(raw_output)
    except (TypeError, ValueError):
        report = {}
    public = report.get("public", {}) if isinstance(report, dict) else {}
    independent = report.get("independent", {}) if isinstance(report, dict) else {}
    failures = sorted(set(FAILED_TEST.findall(public.get("output", ""))))
    return {
        "public_failures": failures,
        "independent_passed": [line for line in independent.get("output", "").splitlines()
                               if line.startswith("PASS ")],
        "fallback_output": None if failures else raw_output.strip()[-1000:],
    }
