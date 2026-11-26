# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture identities in the selected interpreter and in actual model workers."""

from datetime import datetime, timezone
import importlib
import importlib.metadata
import os
from pathlib import Path
import re
import sys

from .common import capture as command, git, read_json, sha256, write_json

MODULES = ("pypto", "pypto.pypto_core", "simpler", "_task_interface", "pypto_serving", "torch")
KEYS = ("repo_commits", "actual_imports", "runtime_native", "compiler_native", "pto_isa", "ptoas",
        "torch_dependencies", "source_identity", "cache_isolation", "allocated_devices", "all_model_workers")


def capture(manifest, label, with_imports=False, loaded_only=False):
    record = {"schema_version": 1, "label": label, "pid": os.getpid(), "ppid": os.getppid(), "python": sys.executable,
              "argv": sys.argv, "sys_path": list(sys.path),
              "sys_prefix": str(Path(sys.prefix).resolve()), "python_version": sys.version,
              "at": datetime.now(timezone.utc).isoformat(), "repositories": {}, "imports": {},
              "environment": {k: os.environ.get(k) for k in (
                  "PYPTO_LIB_ROOT", "PTOAS_ROOT", "TASK_DEVICE", "PYPTO_CACHE", "PYPTO_CACHE_DIR", "PYTHONPATH",
                  "PYPTO_PROG_BUILD_DIR", "PYPTO_RUNTIME_ROOT", "PYPTO_CODEGEN_MAX_WORKERS")}}
    for name, item in manifest["sources"].items():
        path = Path(item["path"])
        record["repositories"][name] = {"commit": git(path, "rev-parse", "HEAD"),
            "status": git(path, "status", "--porcelain=v1"),
            "submodules": git(path, "submodule", "status", "--recursive")}
    if with_imports:
        for name in MODULES:
            try:
                module = sys.modules.get(name) if loaded_only else importlib.import_module(name)
                if module is None:
                    raise ValueError("Module not loaded in this worker")
                path = Path(module.__file__).resolve()
                record["imports"][name] = {"path": str(path), "sha256": sha256(path),
                    "build_commit": getattr(module, "__build_commit__", None)}
            except Exception as exc:
                record["imports"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        runtime = Path(manifest["sources"]["runtime"]["path"])
        isa = runtime / "build/pto-isa"
        record["isa"] = {"path": str(isa), "commit": git(isa, "rev-parse", "HEAD"),
                         "status": git(isa, "status", "--porcelain=v1")}
        metadata = runtime / "build/lib/pto_isa_build.json"
        record["isa_build"] = read_json(metadata) if metadata.is_file() else None
        record["runtime_libraries"] = {
            str(p.relative_to(runtime)): sha256(p) for p in sorted((runtime / "build/lib").rglob("*.so"))}
        ptoas = Path(manifest["run_dir"]) / "ptoas/bin/ptoas"
        record["ptoas"] = {"path": str(ptoas), "sha256": sha256(ptoas),
                           "version": command([ptoas, "--version"], timeout=30)}
        record["torch"] = importlib.metadata.version("torch")
    write_json(manifest["output"], record)
    return record


def under(path, root):
    return bool(path and Path(path).resolve().is_relative_to(Path(root).resolve()))


def checks(manifest, observed, built, expected_worker_count):
    repos = observed.get("repositories", {})
    imports = observed.get("imports", {})
    sources = manifest["sources"]
    result = {key: False for key in KEYS}
    result["repo_commits"] = all(repos.get(k, {}).get("commit") == v["commit"] for k, v in sources.items())
    result["source_identity"] = result["repo_commits"] and all(
        {line.strip() for line in repos[k].get("status", "").splitlines()} <= (
            {"M pypto-lib"} if k == "serving" and manifest["lib_override"] else set())
        for k in sources)
    result["actual_imports"] = observed.get("sys_prefix") == manifest["conda_prefix"] and all(
        under(imports.get(mod, {}).get("path"), sources[src]["path"])
        for mod, src in (("pypto", "pypto"), ("simpler", "runtime"), ("pypto_serving", "serving")))
    result["actual_imports"] &= observed.get("environment", {}).get("PYPTO_LIB_ROOT") == sources["lib"]["path"]
    for key, module in (("runtime_native", "_task_interface"), ("compiler_native", "pypto.pypto_core")):
        item, reference = imports.get(module, {}), built.get("imports", {}).get(module, {})
        result[key] = bool(item.get("sha256") and item.get("sha256") == reference.get("sha256")
                           and item.get("path") == reference.get("path"))
    result["runtime_native"] &= imports.get("_task_interface", {}).get("build_commit") == sources["runtime"]["commit"]
    isa = observed.get("isa", {})
    metadata = observed.get("isa_build") or {}
    result["pto_isa"] = (isa.get("commit") == manifest["pto_isa"] and not isa.get("status")
        and metadata.get("required_commit_from_pin") == manifest["pto_isa"]
        and metadata.get("actual_checkout_commit") == manifest["pto_isa"]
        and bool(observed.get("runtime_libraries"))
        and observed.get("runtime_libraries") == built.get("runtime_libraries"))
    artifacts = metadata.get("runtime_artifacts", {})
    result["pto_isa"] &= metadata.get("schema_version") == 3 and bool(artifacts) and all(
                            item.get("required_commit_from_pin") == manifest["pto_isa"]
                            and item.get("actual_checkout_commit") == manifest["pto_isa"]
                            for item in artifacts.values())
    ptoas, reference = observed.get("ptoas", {}), built.get("ptoas", {})
    version = manifest["ptoas"]["version"].removeprefix("v")
    result["ptoas"] = (bool(ptoas.get("sha256")) and ptoas.get("sha256") == reference.get("sha256")
        and bool(re.search(rf"(?<![\d.]){re.escape(version)}(?![\d.])", ptoas.get("version", ""))))
    result["torch_dependencies"] = observed.get("torch", "").split("+")[0] == manifest["ci"]["torch"].split("==")[-1]
    env = observed.get("environment", {})
    result["cache_isolation"] = (
        env.get("PYPTO_CACHE") == "1" and env.get("PYPTO_CACHE_DIR") == str(Path(manifest["run_dir"]) / "jit/cache")
        and not env.get("PYPTO_PROG_BUILD_DIR") and not env.get("PYPTO_RUNTIME_ROOT"))
    try:
        devices = [int(x) for x in (observed.get("environment", {}).get("TASK_DEVICE") or "").split(",")]
        result["allocated_devices"] = len(devices) == len(set(devices)) == 16 and min(devices) >= 0
    except ValueError:
        pass
    result["all_model_workers"] = expected_worker_count > 0
    return result


def model_worker_roots(records):
    """Distinguish Serving workers from their observed forked device descendants."""
    parents = {item.get("pid"): item.get("ppid") for item in records}
    if (len(parents) != len(records)
            or any(type(pid) is not int or type(ppid) is not int or min(pid, ppid) <= 0
                   for pid, ppid in parents.items())):
        return None
    for pid in parents:
        seen = set()
        while pid in parents:
            if pid in seen:
                return None
            seen.add(pid)
            pid = parents[pid]
    return {pid for pid, ppid in parents.items() if ppid not in parents}


def validate_workers(manifest, root):
    root = Path(root)
    built = read_json(root / "environment/built.json")
    log = (root / "server/server.log").read_text(errors="replace")
    expected = len(re.findall(r"\bWorker ready\b", log))
    workers = sorted((root / "environment/workers").glob("worker-*.json"))
    aggregate = {key: True for key in KEYS}
    observations, records = [], []
    for path in workers:
        observed = read_json(path)
        records.append(observed)
        current = checks(manifest, observed, built, expected)
        observations.append({"file": str(path.relative_to(root)), "pid": observed["pid"],
                             "ppid": observed.get("ppid"), "checks": current})
        for key in KEYS:
            aggregate[key] &= current[key]
    roots = model_worker_roots(records)
    chip_pids = {int(pid) for pid in re.findall(r"\[chip_process pid=(\d+) dev=\d+\] ready", log)}
    observed_pids = {item["pid"] for item in records}
    aggregate["all_model_workers"] = (roots is not None and expected > 0 and len(roots) == expected
                                      and chip_pids <= observed_pids - roots)
    if not workers:
        aggregate = dict.fromkeys(KEYS, False)
    write_json(root / "identity-checks.json", {"checks": aggregate, "workers": observations,
                                               "expected_workers": expected,
                                               "model_worker_pids": sorted(roots or []),
                                               "device_process_pids": sorted(chip_pids)})
    return aggregate
