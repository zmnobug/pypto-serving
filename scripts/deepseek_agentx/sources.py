# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resolve the supported lib CI contract and freeze independent source checkouts."""

from __future__ import annotations

import platform
from pathlib import Path
import re
import subprocess

from .common import RegressionError, capture, git, sha256, write_json


def literal(value):
    value = str(value)
    if "${{" in value or "\n" in value:
        raise RegressionError("BLOCKED_COMPATIBILITY", f"Unsupported dynamic CI input: {value}")
    return value


def parse_ci(root, job_name):
    workflow = root / ".github/workflows/ci.yml"
    setup = root / ".github/actions/setup-ci-job/action.yml"
    serving = root / ".github/actions/pypto-serving-tests/action.yml"
    try:
        import yaml

        steps = yaml.safe_load(workflow.read_text())["jobs"][job_name]["steps"]
        setup_action = yaml.safe_load(setup.read_text())
        serving_action = yaml.safe_load(serving.read_text())
        setup_calls = [s for s in steps if s.get("uses") == "./.github/actions/setup-ci-job"]
        serving_calls = [s for s in steps if s.get("uses") == "./.github/actions/pypto-serving-tests"]
        if len(setup_calls) != 1 or len(serving_calls) != 1:
            raise ValueError("Expected one setup and one serving action")
        effective = {k: v.get("default", "") for k, v in setup_action["inputs"].items()}
        effective.update(setup_calls[0].get("with", {}))
        effective = {k: literal(v) for k, v in effective.items()}
        if effective["runtime-platform"] != "a2a3" or effective["needs-device"] != "true":
            raise ValueError("CI job no longer targets A2/A3 devices")
        if effective["install-torch-npu"] != "false":
            raise ValueError("This adapter supports the CPU-Torch DSpark CI contract only")
        if serving_calls[0]["with"]["model-name"] != "deepseek-dspark":
            raise ValueError("CI job is not the DSpark guard")
        checkouts = [s for s in serving_action["runs"]["steps"]
                     if s.get("uses", "").startswith("actions/checkout@")]
        if len(checkouts) != 1:
            raise ValueError("Serving checkout contract changed")
        checkout = checkouts[0]["with"]
        setup_text, serving_text = setup.read_text(), serving.read_text()
        if "https://github.com/hw-native-sys/pypto.git" not in setup_text:
            raise ValueError("PyPTO source URL contract changed")
        if "Override pypto-serving submodule with pypto-lib under test" not in serving_text:
            raise ValueError("Lib-under-test override contract changed")
        if "cp310-cp310-manylinux_2_34_" not in setup_text:
            raise ValueError("PTOAS wheel ABI contract changed")
        return {
            "pypto_ref": effective["pypto-ref"] or "HEAD",
            "serving_url": "https://github.com/" + literal(checkout["repository"]) + ".git",
            "serving_ref": literal(checkout.get("ref", "HEAD")),
            "torch": effective["torch-version"], "python": "3.10",
            "effective_setup_inputs": effective,
            "files": [str(p.relative_to(root)) for p in (workflow, setup, serving)],
        }
    except (KeyError, TypeError, ValueError, ImportError, OSError) as exc:
        raise RegressionError("BLOCKED_COMPATIBILITY", f"Unsupported lib CI contract: {exc}") from exc


def clone(url, ref, destination):
    """Never reset/clean a user tree; destination must be new."""
    if destination.exists():
        raise RegressionError("BLOCKED_SETUP", f"Checkout already exists: {destination}")
    try:
        capture(["git", "clone", "--no-checkout", url, destination], timeout=900)
        git(destination, "fetch", "origin", ref)
        commit = git(destination, "rev-parse", "FETCH_HEAD")
        git(destination, "checkout", "--detach", commit)
        return commit
    except (subprocess.SubprocessError, OSError) as exc:
        raise RegressionError("BLOCKED_FETCH", f"Unable to fetch {url} {ref}: {exc}") from exc


def gitlink(root, path):
    fields = git(root, "ls-tree", "HEAD", path).split()
    if len(fields) != 4 or fields[0] != "160000":
        raise RegressionError("BLOCKED_COMPATIBILITY", f"Missing {path} gitlink in {root}")
    return fields[2]


def source_record(path):
    lines = git(path, "show", "-s", "--format=%H%n%cI%n%s").splitlines()
    return {"path": str(path), "commit": lines[0], "date": lines[1], "subject": lines[2],
            "status": git(path, "status", "--porcelain=v1"),
            "submodules": git(path, "submodule", "status", "--recursive"),
            "diff": git(path, "diff", "--binary", "--ignore-submodules=all"),
            "staged_diff": git(path, "diff", "--cached", "--binary", "--ignore-submodules=all")}


def snapshot_sources(manifest):
    records = {}
    for name, expected in manifest["sources"].items():
        record = source_record(Path(expected["path"]))
        statuses = {line.strip() for line in record["status"].splitlines() if line.strip()}
        if name == "serving" and manifest["lib_override"]:
            statuses = {line for line in statuses if line != "M pypto-lib"}
        if record["commit"] != expected["commit"] or statuses or record["diff"] or record["staged_diff"]:
            raise RegressionError("BLOCKED_SETUP", f"Source identity changed: {name}: {record['status']}")
        records[name] = record
    return records


def resolve(root, profile):
    checkouts = root / "checkouts"
    checkouts.mkdir()
    repos = profile["repositories"]
    lib_ci = checkouts / "lib-ci"
    clone(repos["lib"], "HEAD", lib_ci)
    ci = parse_ci(lib_ci, profile["ci_job"])
    ci_dir = root / "ci-sources"
    ci_dir.mkdir(exist_ok=True)
    for name in ci["files"]:
        target = ci_dir / name.removeprefix(".github/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((lib_ci / name).read_bytes())
    serving, pypto = checkouts / "serving", checkouts / "pypto"
    clone(ci["serving_url"], ci["serving_ref"], serving)
    clone(repos["pypto"], ci["pypto_ref"], pypto)
    runtime_sha = gitlink(pypto, "runtime")
    try:
        capture(["git", "-C", pypto, "submodule", "update", "--init", "--recursive"], timeout=1800)
        if git(pypto / "runtime", "rev-parse", "HEAD") != runtime_sha:
            raise ValueError("runtime checkout differs from PyPTO gitlink")
        original_lib = gitlink(serving, "pypto-lib")
        if (serving / "pypto-lib").exists():
            (serving / "pypto-lib").rmdir()
        clone(repos["lib"], git(lib_ci, "rev-parse", "HEAD"), serving / "pypto-lib")
        # Initialize other submodules without reverting the deliberate lib override.
        modules = serving / ".gitmodules"
        if modules.is_file():
            paths = capture(["git", "config", "-f", modules, "--get-regexp", r"^submodule\..*\.path$"])
            for line in paths.splitlines():
                path = line.split(None, 1)[1]
                if path != "pypto-lib":
                    capture(["git", "-C", serving, "submodule", "update", "--init", "--recursive", "--", path],
                            timeout=1800)
    except (subprocess.SubprocessError, ValueError) as exc:
        raise RegressionError("BLOCKED_FETCH", f"Unable to initialize pinned submodules: {exc}") from exc
    clone(repos["agentx"], "HEAD", checkouts / "agentx-harness")
    versions_file = pypto / "toolchain/versions.env"
    versions = dict(re.findall(r"^([A-Z0-9_]+)=([^\s#]+)", versions_file.read_text(), re.MULTILINE))
    arch = platform.machine()
    try:
        ptoas = {"version": versions["PTOAS_VERSION"], "sha256": versions[f"PTOAS_SHA256_{arch.upper()}"],
                 "architecture": arch}
        isa = (pypto / "runtime/pto_isa.pin").read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", isa) or not re.fullmatch(r"[0-9a-f]{64}", ptoas["sha256"]):
            raise ValueError("Invalid toolchain identity")
    except (KeyError, ValueError) as exc:
        raise RegressionError("BLOCKED_COMPATIBILITY", f"Unsupported toolchain pins: {exc}") from exc
    (ci_dir / "versions.env").write_bytes(versions_file.read_bytes())
    (ci_dir / "pto_isa.pin").write_text(isa + "\n")
    paths = {"lib_ci": lib_ci, "serving": serving, "lib": serving / "pypto-lib", "pypto": pypto,
             "runtime": pypto / "runtime", "agentx": checkouts / "agentx-harness"}
    sources = {name: {**source_record(path), "url": git(path, "remote", "get-url", "origin")}
               for name, path in paths.items()}
    for name, path in paths.items():
        if (path / "AGENTS.md").is_file():
            (ci_dir / f"{name}-AGENTS.md").write_bytes((path / "AGENTS.md").read_bytes())
    result = {"sources": sources, "ci": ci, "ptoas": ptoas, "pto_isa": isa,
              "lib_gitlink": original_lib, "lib_override": sources["lib"]["commit"] != original_lib,
              "ci_files": {str(p.relative_to(ci_dir)): sha256(p) for p in ci_dir.rglob("*") if p.is_file()}}
    write_json(root / "version-plan.json", result)
    snapshot_sources(result)
    return result
