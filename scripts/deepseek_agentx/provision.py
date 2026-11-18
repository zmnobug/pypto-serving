# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare one existing Conda environment from the frozen CI matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import time

from .common import RegressionError, capture, clean_environment, git, run_command, sha256, write_json


def conda_environment(name):
    executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not executable:
        raise RegressionError("BLOCKED_SETUP", "conda is not available; provide it on PATH")
    envs = json.loads(capture([executable, "env", "list", "--json"]))["envs"]
    requested = Path(name).expanduser()
    matches = [Path(p).resolve() for p in envs
               if Path(p).name == name or (requested.is_absolute() and Path(p).resolve() == requested.resolve())]
    if len(matches) != 1 or not (matches[0] / "bin/python").is_file():
        raise RegressionError("BLOCKED_SETUP", f"Expected one existing Conda environment for {name!r}")
    prefix = matches[0]
    observed = capture([prefix / "bin/python", "-c", "import sys; print(sys.prefix)"])
    if Path(observed).resolve() != prefix:
        raise RegressionError("BLOCKED_SETUP", "Conda interpreter prefix does not match requested environment")
    base = Path(capture([executable, "info", "--base"]))
    return prefix, base / "etc/profile.d/conda.sh", executable


def environment_busy(prefix):
    users = []
    for entry in Path("/proc").glob("[0-9]*"):
        if int(entry.name) == os.getpid():
            continue
        try:
            if (entry / "exe").resolve().is_relative_to(prefix):
                users.append(int(entry.name))
        except (OSError, RuntimeError):
            continue
    return users


def model_identity(model):
    try:
        config = json.loads((model / "config.json").read_text())
        index_path = model / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
        if not shards:
            raise ValueError("Empty safetensors index")
        weights = {}
        for name in shards:
            path = (model / name).resolve()
            if not path.is_relative_to(model.resolve()):
                raise ValueError("Weight shard is outside model directory")
            stat = path.stat()
            if stat.st_size == 0:
                raise ValueError(f"Empty weight shard: {name}")
            weights[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        tokenizer_files = sorted(p for p in model.glob("*token*") if p.is_file())
        if not (model / "tokenizer_config.json").is_file() or not tokenizer_files:
            raise ValueError("Missing tokenizer files")
        metadata = {p.name: sha256(p) for p in [model / "config.json", index_path, *tokenizer_files]}
        return {"path": str(model), "model_type": config.get("model_type"),
                "quantization_config": config.get("quantization_config"), "metadata_sha256": metadata,
                "weights": weights, "weight_identity_method": "path-size-mtime; not content hashes"}
    except (OSError, ValueError, KeyError) as exc:
        raise RegressionError("BLOCKED_SETUP", f"Incomplete model/tokenizer directory {model}: {exc}") from exc


def tokenizer_identity(tokenizer):
    tokenizer = Path(tokenizer)
    if not tokenizer.is_dir() or not (tokenizer / "tokenizer_config.json").is_file():
        raise RegressionError("BLOCKED_SETUP", f"Tokenizer directory is incomplete: {tokenizer}")
    files = sorted(p for p in tokenizer.glob("*token*") if p.is_file())
    if not files:
        raise RegressionError("BLOCKED_SETUP", f"Tokenizer directory has no tokenizer files: {tokenizer}")
    return {"path": str(tokenizer.resolve()), "metadata_sha256": {p.name: sha256(p) for p in files}}


def identity_fingerprint(value):
    """Normalize host paths out of model/tokenizer identities for replay checks."""
    if isinstance(value, dict):
        return {key: identity_fingerprint(item) for key, item in value.items()
                if key not in {"path", "mtime_ns"}}
    if isinstance(value, list):
        return [identity_fingerprint(item) for item in value]
    return value


def activation_preamble(manifest):
    reset = """while IFS= read -r key; do
  case "$key" in
    PYPTO_*|PTOAS_*|PTO_ISA_*|SIMPLER_*|PYTHONPATH|PYTHONHOME|VIRTUAL_ENV|ASCEND_GLOBAL_LOG_LEVEL|TASKQUEUE_IGNORE_*)
      unset "$key" ;;
  esac
done < <(compgen -e)
"""
    # Conda env config vars and activation hooks can restore a previous checkout.
    # Reset again after activation, before CANN contributes its own Python paths.
    return (reset + f"source {shlex.quote(manifest['conda_sh'])} >/dev/null\n"
            + f"conda activate {shlex.quote(manifest['conda_prefix'])}\n" + reset
            + f"source {shlex.quote(manifest['cann_env'])} >/dev/null\n")


def activation_script(manifest, env):
    text = "#!/usr/bin/env bash\nset -eo pipefail\n" + activation_preamble(manifest)
    keys = ("PYPTO_LIB_ROOT", "PYPTO_ROOT", "PTOAS_ROOT", "PYPTO_CACHE", "PYPTO_CACHE_DIR",
            "ASCEND_PROCESS_LOG_PATH", "PYTHONPATH", "PATH", "LD_LIBRARY_PATH",
            "TORCH_DEVICE_BACKEND_AUTOLOAD", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
            "CMAKE_BUILD_PARALLEL_LEVEL", "MAX_JOBS", "PYPTO_BUILD_JOBS", "PYPTO_TEST_JOBS",
            "PYPTO_CODEGEN_MAX_WORKERS")
    for key in keys:
        text += f"export {key}={shlex.quote(env[key])}\n"
    return text


def activation_environment(manifest):
    args = ["bash", "-c", "set -e\n" + activation_preamble(manifest) + "exec env -0"]
    output = capture(args, env=clean_environment())
    env = dict(item.split("=", 1) for item in output.split("\0") if "=" in item)
    root = Path(manifest["run_dir"])
    sources = manifest["sources"]
    env.update(PYPTO_LIB_ROOT=sources["lib"]["path"], PYPTO_ROOT=sources["pypto"]["path"],
               PTOAS_ROOT=str(root / "ptoas"), PYPTO_CACHE="1", PYPTO_CACHE_DIR=str(root / "jit/cache"),
               ASCEND_PROCESS_LOG_PATH=str(root / "server/ascend"),
               CMAKE_BUILD_PARALLEL_LEVEL="2", MAX_JOBS="2", PYPTO_BUILD_JOBS="2", PYPTO_TEST_JOBS="2",
               PYPTO_CODEGEN_MAX_WORKERS="2",
               TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("PTO_ISA_ROOT", None)  # The runtime's managed, pinned checkout is authoritative.
    cann_root = Path(manifest["cann_env"]).parent
    env["PATH"] = os.pathsep.join([str(prefix) for prefix in (
        root / "ptoas/bin", cann_root / "tools/hcc/bin")]) + os.pathsep + env["PATH"]
    ptoas_site = root / "ptoas/lib/python3.10/site-packages"
    server_site = Path(manifest["conda_prefix"]) / "lib/python3.10/site-packages"
    cann_python = [path for path in env.get("PYTHONPATH", "").split(os.pathsep) if path]
    env["PYTHONPATH"] = os.pathsep.join([str(server_site), str(ptoas_site), *cann_python])
    env["LD_LIBRARY_PATH"] = str(Path(manifest["conda_prefix"]) / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def project(path):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with (path / "pyproject.toml").open("rb") as source:
        return tomllib.load(source)


def supports_version(specifier, version):
    from packaging.specifiers import SpecifierSet
    return bool(specifier.strip()) and SpecifierSet(specifier).contains(version)


def exact_torch(requirement):
    match = re.fullmatch(r"torch==([0-9]+(?:\.[0-9]+)+)", requirement.strip())
    if not match:
        raise RegressionError("BLOCKED_COMPATIBILITY", "CI torch requirement is no longer an exact version")
    return match.group(1)


def requirement_name(requirement):
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()


def download_checked(url, destination, expected_sha256, execute, env, cache=None):
    # Use the host's download tool so uv-managed Python does not need a second CA setup.
    cached = Path(cache) / expected_sha256 / destination.name if cache is not None else None
    if cached is not None and cached.is_file() and sha256(cached) == expected_sha256:
        shutil.copyfile(cached, destination)
        return "verified-cache"
    execute(["curl", "--http1.1", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
             "--retry-all-errors", "--connect-timeout", "30", "--output", destination, url], env=env)
    if sha256(destination) != expected_sha256:
        raise RegressionError("BLOCKED_SETUP", "PTOAS wheel SHA256 differs from the compiler pin")
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_name(f".{cached.name}.{os.getpid()}.tmp")
        shutil.copyfile(destination, temporary)
        temporary.replace(cached)
    return "download"


def rebuild_projects(projects, runtime, python, execute, env, log):
    pip = [python, "-m", "pip"]
    # PEP 660 import finders from an old checkout can shadow the new build helpers.
    # --force-reinstall alone removes them only after the replacement wheel is built.
    execute([*pip, "uninstall", "-y", *(data["project"]["name"] for _, data in projects)], log, env=env)
    for path, _ in projects:
        command = [*pip, "install", "--no-deps", "--no-build-isolation", "--force-reinstall", "-e", path]
        if path == runtime:
            command.extend(["--config-settings=build.targets=build_package_a2a3"])
        execute(command, log, env=env)


def dependency_check_environment(env):
    result = dict(env)
    # CANN exports optional tools on PYTHONPATH; they are not Conda distributions.
    result.pop("PYTHONPATH", None)
    return result


def prepare(manifest):
    root = Path(manifest["run_dir"])
    prefix = Path(manifest["conda_prefix"])
    python = prefix / "bin/python"
    env = activation_environment(manifest)
    log = root / "environment/install.log"
    budget = manifest["profile"]["budgets"]["setup_seconds"]
    deadline = time.monotonic() + budget

    def execute(args, target=log, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RegressionError("BLOCKED_SETUP", "Overall setup budget exhausted")
        kwargs.pop("timeout", None)
        run_command(args, target, timeout=remaining, **kwargs)

    pip = [python, "-m", "pip"]
    py_version = capture([python, "-c", "import sys; print('.'.join(map(str,sys.version_info[:3])))"], env=env)
    if not py_version.startswith(manifest["ci"]["python"] + "."):
        raise RegressionError("BLOCKED_COMPATIBILITY", f"CI PTOAS wheel requires Python 3.10, found {py_version}")
    pypto = Path(manifest["sources"]["pypto"]["path"])
    runtime, serving = pypto / "runtime", Path(manifest["sources"]["serving"]["path"])
    projects = [(path, project(path)) for path in (pypto, runtime, serving)]
    for path, data in projects:
        if not supports_version(data["project"].get("requires-python", ""), py_version):
            raise RegressionError("BLOCKED_COMPATIBILITY", f"{path.name} does not support Python {py_version}")
    for compiler in ("ccec", "aarch64-target-linux-gnu-g++"):
        if not shutil.which(compiler, path=env["PATH"]):
            raise RegressionError("BLOCKED_SETUP", f"Missing CANN compiler: {compiler}")
    execute([*pip, "freeze"], root / "environment/pip-before.txt", env=env)
    execute([manifest["conda_executable"], "list", "--explicit", "--prefix", prefix],
                root / "environment/conda-before.txt", env=env)
    constraints = root / "environment/constraints.txt"
    exact_torch(manifest["ci"]["torch"])
    constraints.write_text(manifest["ci"]["torch"] + "\n" +
                           ((pypto / "build-constraints.txt").read_text()
                            if (pypto / "build-constraints.txt").is_file() else ""))
    env["PIP_CONSTRAINT"] = str(constraints)
    env["PIP_INDEX_URL"] = manifest["index_url"]
    # Dependencies use PyPI; the Torch-specific index only supplies Torch itself.
    execute([*pip, "install", "packaging", "tomli", "pytest<9.2", "pytest-xdist", "PyYAML"], log,
                env=env, timeout=budget)
    torch_dir = root / "environment/torch"
    torch_dir.mkdir()
    execute([*pip, "download", "--no-deps", manifest["ci"]["torch"], "--dest", torch_dir,
             "--index-url", "https://download.pytorch.org/whl/cpu"], env=env)
    torch_wheels = list(torch_dir.glob("torch-*.whl"))
    if len(torch_wheels) != 1:
        raise RegressionError("BLOCKED_SETUP", "Expected exactly one CPU Torch wheel")
    execute([*pip, "install", "--force-reinstall", torch_wheels[0]], env=env)
    # A reused environment may still contain a torch-npu wheel from an older matrix.
    execute([*pip, "uninstall", "-y", "torch-npu"], env=env)
    write_json(root / "environment/torch-wheel.json", {"name": torch_wheels[0].name,
                                                        "sha256": sha256(torch_wheels[0])})
    dependencies = []
    for _, data in projects:
        dependencies.extend(data["build-system"]["requires"])
        dependencies.extend(req for req in data["project"].get("dependencies", [])
                            if requirement_name(req) not in {"torch", "pypto", "simpler"})
    execute([*pip, "install", *sorted(set(dependencies)), "-r", serving / "pypto_serving/requirements-runtime.txt"],
                log, env=env, timeout=budget)
    ptoas = manifest["ptoas"]
    version = ptoas["version"].removeprefix("v")
    wheel_name = f"ptoas-{version}-cp310-cp310-manylinux_2_34_{ptoas['architecture']}.whl"
    wheel = root / "environment" / wheel_name
    url = f"https://github.com/hw-native-sys/PTOAS/releases/download/{ptoas['version']}/{wheel_name}"
    wheel_source = download_checked(url, wheel, ptoas["sha256"], execute, env,
                                    Path.home() / ".cache/deepseek-agentx/downloads")
    write_json(root / "environment/ptoas-wheel.json", {"url": url, "sha256": sha256(wheel), "source": wheel_source})
    print(f"PTOAS wheel: {wheel_source}; pinned SHA256 verified", flush=True)
    execute([python, "-m", "venv", root / "ptoas"], log, env=env)
    execute([root / "ptoas/bin/python", "-m", "pip", "install", "--no-deps", wheel], log, env=env)
    rebuild_projects(projects, runtime, python, execute, env, log)
    # Prevent the test runner's working directory from shadowing this editable install.
    tests = ["tests/unit/serving", "tests/unit/model/deepseek_dspark/test_dspark_model.py"]
    if any(not (serving / path).exists() for path in tests):
        raise RegressionError("BLOCKED_COMPATIBILITY", "Serving CPU test paths changed; update the adapter")
    execute([python, "-m", "pytest", "-q", "-n", "2", *tests], root / "environment/cpu-tests.log",
                cwd=serving, env=env, timeout=budget)
    execute([*pip, "check"], root / "environment/pip-check.log", env=dependency_check_environment(env))
    execute([*pip, "freeze"], root / "environment/pip-after.txt", env=env)
    agentx = Path(manifest["sources"]["agentx"]["path"])
    client_env = clean_environment()
    execute(["uv", "venv", "--python", manifest["profile"]["agentx"]["python"], root / ".venv-agentx"],
                root / "environment/agentx-install.log", env=client_env)
    execute(["uv", "pip", "install", "--python", root / ".venv-agentx/bin/python", "-e", agentx],
                root / "environment/agentx-install.log", env=client_env, timeout=budget)
    execute(["uv", "pip", "freeze", "--python", root / ".venv-agentx/bin/python"],
                root / "environment/agentx-pip-freeze.txt", env=client_env)
    execute([root / ".venv-agentx/bin/aiperf", "--help"], root / "environment/agentx-help.txt", env=client_env)
    # Fresh independent builds establish the expected native fingerprints. Worker
    # observations must match these, not define their own expected values.
    tool = Path(manifest["driver_root"]) / "scripts/run_deepseek_agentx_regression.py"
    execute([python, tool, "identity", "--run-dir", root, "--output", root / "environment/built.json", "--with-imports"],
                root / "environment/identity.log", cwd=root, env=env)
    from .identity import checks
    built = json.loads((root / "environment/built.json").read_text())
    validated = checks(manifest, built, built, 1)
    required = {key: value for key, value in validated.items()
                if key not in {"allocated_devices", "all_model_workers"}}
    write_json(root / "environment/build-checks.json", required)
    if not all(required.values()):
        raise RegressionError("BLOCKED_SETUP", f"Fresh build/import identity failed: {required}")
    server_help = capture([python, "-m", "pypto_serving.cli", "--help"], cwd=serving, env=env, timeout=60)
    (root / "environment/server-help.txt").write_text(server_help)
    from .engine import server_command, client_command
    client_help = capture([root / ".venv-agentx/bin/aiperf", "profile", "--help"], env=client_env, timeout=60)
    (root / "environment/agentx-profile-help.txt").write_text(client_help)
    for command, help_text in ((server_command(manifest), server_help), (client_command(manifest, 1), client_help)):
        absent = [arg for arg in command if arg.startswith("--") and arg not in help_text]
        if absent:
            raise RegressionError("BLOCKED_COMPATIBILITY", f"CLI flags no longer supported: {absent}")
    (root / "activate-run.sh").write_text(activation_script(manifest, env))
    execute(["bash", "-n", root / "activate-run.sh"], root / "environment/script-check.log")
    write_json(root / "environment/toolchain.json", {
        "ptoas_version": capture([root / "ptoas/bin/ptoas", "--version"], env=env),
        "wheel": wheel_name, "wheel_sha256": sha256(wheel), "runtime_pin": git(runtime, "rev-parse", "HEAD"),
        "compilers": {name: {"path": str(Path(shutil.which(name, path=env["PATH"])).resolve()),
                              "sha256": sha256(shutil.which(name, path=env["PATH"])),
                              "version": capture([shutil.which(name, path=env["PATH"]), "--version"], env=env)}
                      for name in ("ccec", "aarch64-target-linux-gnu-g++")},
    })
