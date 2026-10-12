# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Render a self-contained, offline observation report for one Agent repair run."""

import argparse
import json
from pathlib import Path

from analyze_runs import analyze_run


def read_optional_list(path: Path) -> list:
    if not path.is_file():
        return []
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return value


def render_report(run_dir: Path, output: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    if run_dir.is_file():
        run_dir = run_dir.parent
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict) or not isinstance(summary.get("result"), dict):
        raise ValueError(f"Not an Agent repair run: {summary_path}")

    trajectory_path = run_dir / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.is_file() else {}
    payload = {
        "run_name": run_dir.name,
        "summary": summary,
        "analysis": analyze_run(run_dir),
        "requests": read_optional_list(run_dir / "request_events.json"),
        "actions": read_optional_list(run_dir / "action_events.json"),
        "context_events": trajectory.get("info", {}).get("context_management", {}).get("requests", []),
    }
    # JSON sits in a script element, so '<' must not be able to close that element.
    data = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("report_template.html").read_text()
    marker = "__REPORT_DATA__"
    if template.count(marker) != 1:
        raise ValueError("Report template must contain exactly one data marker")
    output = Path(output) if output is not None else run_dir / "observation_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace(marker, data))
    return output.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Repair run directory or its summary.json")
    parser.add_argument("--output", type=Path, help="HTML destination (default: RUN/observation_report.html)")
    args = parser.parse_args()
    try:
        print(render_report(args.run, args.output))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f"render_report.py: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
