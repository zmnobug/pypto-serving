# DeepSeek + AgentX automatic regression

This is an executable regression harness. It resolves the current upstream lib
CI matrix, builds it in an existing service Conda environment, runs a 16-device
DeepSeek DSpark service and AgentX, then stops its own tasks and archives the
result. The implementation does not modify the model or Serving product code.

## Run once

Run from the Serving repository root on the NPU host. Requirements:

- An existing, idle Python 3.10 Conda environment; its Python version is not changed.
- CANN with `set_env.sh`, `ccec` and `aarch64-target-linux-gnu-g++`.
- `git`, `uv`, `curl`, `task-submit`, network access to the source repositories, PyPI,
  CPU Torch wheels, PTOAS releases, and the AgentX dataset.
- A local model directory with config, safetensors index, every weight shard
  and tokenizer files. A separate local tokenizer directory is optional.
- Permission through the normal task queue to allocate 16 devices together,
  and enough disk space for separate source builds, AgentX reports and kernels.

Prepare a small controller environment, separate from the service environment:

```bash
uv venv --python 3.12 artifacts/regression-driver
uv pip install --python artifacts/regression-driver/bin/python \
  -r scripts/deepseek_agentx/requirements.txt
artifacts/regression-driver/bin/python scripts/run_deepseek_agentx_regression.py plan
```

Run with your host paths and existing Conda environment name or absolute prefix:

```bash
artifacts/regression-driver/bin/python scripts/run_deepseek_agentx_regression.py full \
  --work-root "$PWD" \
  --conda-env zm_pypto \
  --model /path/to/dsv4-flash-dspark-w8a8 \
  --cann-root /path/to/Ascend/cann
```

To wait indefinitely for the 16-device allocation and remove the queue task's
runtime limit, append `--task-timeout 0 --task-max-time 0`. The runner then
submits the exact `task-submit --timeout 0 --max-time 0` flags and disables its
matching queue/startup/client watchdogs. Cleanup and explicit fault handling
remain active.

Positive `--task-max-time` values must cover the profile's complete server
budget, including startup, queueing, client execution and cleanup.

To replay the exact source and toolchain matrix from an earlier run, pass its
manifest to a new run. The model and CANN paths may be different, but the model
and tokenizer metadata must match the frozen manifest:

```bash
artifacts/regression-driver/bin/python scripts/run_deepseek_agentx_regression.py full \
  --replay-manifest artifacts/dspark-regression/KNOWN_RUN/manifest.json \
  --work-root "$PWD" --conda-env zm_pypto \
  --model /path/to/dsv4-flash-dspark-w8a8 \
  --cann-root /path/to/Ascend/cann
```

Compare two retained runs without starting a service:

```bash
artifacts/regression-driver/bin/python scripts/run_deepseek_agentx_regression.py compare \
  --baseline artifacts/dspark-regression/KNOWN_RUN \
  --candidate artifacts/dspark-regression/NEW_RUN
```

The comparison reports source and toolchain changes, workload/model drift,
outcome and first-fault changes, and a single changed component as a bisect
candidate. It deliberately does not claim that a changed component is the root
cause; both runs must use the same workload and evidence before drawing that
conclusion.

`--tokenizer` defaults to `--model`. `--port` defaults to 8232. General Python
packages use the Tsinghua index by default (`--index-url` changes it); the CPU
Torch wheel and pinned PTOAS wheel come from their respective upstream sources.
PTOAS downloads use the host's `curl` with HTTP/1.1 and verify the pinned SHA256.
Release wheels are cached under `~/.cache/deepseek-agentx/downloads` by checksum;
every reuse is rehashed, and corrupt cache entries require a fresh download.
This cache contains upstream release packages, not locally compiled source binaries.
Each run records the wheel origin and installs its own PTOAS environment. The runner
inherits the caller's proxy settings when the host requires them.
Git downloads use HTTP/1.1, including recursive submodules, and retain transport
diagnostics in the failure report.
Checkouts and submodules use shallow fetches and still verify every selected SHA;
inactive Git transfers have a bounded low-speed timeout.
The command prints the run directory immediately and emits progress heartbeats.

**The full command updates packages and editable installs in the selected Conda
environment.** Before changing it, the runner saves Conda/pip package inventories
and holds both a workspace lock and an environment-prefix lock through cleanup.
Before rebuilding local projects it uninstalls the old PyPTO, simpler and Serving
distributions, so their editable import finders cannot shadow the new build helpers.
`pip check` validates the Conda environment with CANN's external `PYTHONPATH`
removed; optional CANN analyzer packages remain outside that dependency check.
It refuses an environment used by another running interpreter. Use a dedicated
regression environment. Do not delete an old run's checkout while the environment
still has editable installs referring to it. Preparation is not transactional;
a failed build can leave updated packages, so inspect the retained inventories
and install logs before reusing the environment.
Activation clears old runtime variables both before and after Conda activation:
Conda config variables and hooks must not restore a previous source path or
`PYPTO_PROG_BUILD_DIR`. Build and worker use the same activation preamble, and
host/worker identity checks require this run's JIT cache without an explicit
rebuild directory. Build/codegen parallelism is bounded to two jobs.

## Version selection and workload

The runner fetches lib's remote default branch and reads its `serving-dspark`
job, composite actions, explicit inputs and input defaults. It freezes each SHA
once, and saves the original CI files and relevant repository instructions.

| Component | Source |
| --- | --- |
| lib under test | Fetched remote default branch |
| Serving | Repository/ref selected by lib CI |
| PyPTO | Effective CI `pypto-ref`, or fetched remote HEAD if empty |
| simpler runtime | Selected PyPTO runtime gitlink |
| PTOAS | Selected PyPTO versions file, architecture and wheel SHA256 |
| PTO-ISA | Selected runtime pin and managed build metadata |
| Torch | Exact effective CI version, CPU wheel |
| AgentX | Fetched remote default branch; commit recorded separately |

A deliberate lib-under-test override is recorded and is not reverted by a
submodule update. Existing user source trees and local patches are not used as
upstream sources. Unknown CI structures, dynamic inputs, unsupported wheel ABI,
changed command flags or incompatible builds produce a blocked result; the
runner does not substitute an older known-good version. The current adapter
supports the Python 3.10 / CPU-Torch / A2-A3 contract and removes a leftover
`torch-npu` package when applying that contract.

The committed workload is [full.json](../../tests/regression/deepseek_agentx/full.json):
16 devices allocated by `task-submit`, DP4/TP4/EP16, DSpark K7, prefix cache on,
max context 310000, max sequences 16, batch tokens 10240, chunk threshold 8192,
AgentX concurrency 4, four warmup requests, 1800 formal seconds and seed 20260707.
Server and client commands are generated from this profile. The queue whitelist
and device grouping rules are honored. No device IDs or machine model paths
are embedded in the implementation.

Queue, setup, startup and client budgets are separate. The defaults allow 1 hour
of queueing per submission, 4 hours of preparation, 90 minutes of service startup
and 75 minutes for AgentX configuration, warmup, load, draining and export. The
server task allows four hours, including the client's queue wait, and the CI job
allows twelve hours for fetching, preparation, queueing, execution and cleanup.
A warning or a request exceeding 30 seconds does not end the run. A definite
device or worker fault retains 15 seconds for log flushing before task cleanup;
the runner does not retry a failing round automatically.

## Evidence and acceptance

Each run creates `artifacts/dspark-regression/<run-id>/`:

| Artifact | Purpose |
| --- | --- |
| `manifest.json`, `version-plan.json`, `ci-sources/` | Frozen versions, dates, original CI and toolchain pins |
| `driver/` | Frozen harness code and recorded file hashes |
| `environment/` | Package inventories, build/test logs, native identities and compiler fingerprints |
| `server/server.log`, `server/ascend/` | Service and available CANN host/device logs from all allocated devices |
| `agentx/report/`, `agentx/console.log` | Aggregate report, request records and client output |
| `replay/` | Original HTTP bodies, body hashes and client-to-server request ID mapping |
| `jit/`, `generated-artifacts.json` | Per-run cache and copied default kernel build outputs |
| `events.jsonl`, `first-fault.json`, task records | Timeline, first observed fault and exact submitted tasks |
| `shutdown.json`, `finish-reason.json` | Task/port/process checks and controller termination reason |
| `result.json`, `result.md` | Machine-readable and human-readable result |

A small opt-in observer records already-loaded modules inside the actual model
worker and its forked device processes. Recorded parent IDs distinguish these
process roles; every observation and every logged device process is verified.
It checks the selected interpreter, source SHAs, imported paths, native
hashes, runtime build commit, PTO-ISA build provenance and PTOAS version against
the fresh build. An HTTP observer on localhost forwards streaming responses and
joins AgentX's `X-Request-ID` with Serving's response ID and usage. It also forwards
readiness probes. These observers add some overhead; performance comparisons
must use a separately controlled benchmark.

`PASS_STABILITY` requires a natural AgentX exit 0, the full formal duration,
complete warmup/formal records, no errors or empty outputs, positive actual
prefix-cache hits, unique per-request ID joins and matching token usage,
verified worker/native identities, retained logs/kernels and verified cleanup.
Missing evidence, cancellation, an unrecognized report schema or partial results
cannot pass. The current report adapter supports AgentX schema 1.4.
A source or dependency change alone is never counted as a successful device run.

Results distinguish `FAIL_DEVICE`, `FAIL_SERVER`, `FAIL_REQUESTS`, `INCOMPLETE`,
`BLOCKED_*` and `SKIPPED_BUSY`. A positive result proves stability under this
workload; cached/cold numerical equivalence and performance comparison are
explicitly `NOT_RUN`. Model weight identity uses file paths, sizes and mtimes,
plus metadata/tokenizer hashes, rather than hashing the entire checkpoint.

Reassess retained evidence or stop only this run's tasks:

```bash
artifacts/regression-driver/bin/python scripts/run_deepseek_agentx_regression.py \
  assess --run-dir artifacts/dspark-regression/RUN_ID
artifacts/regression-driver/bin/python scripts/run_deepseek_agentx_regression.py \
  cleanup --run-dir artifacts/dspark-regression/RUN_ID
```

Cleanup never uses broad process-name matching. It cancels/kills only recorded
queue tasks, including submissions recovered by their unique run-script command.
A verified shutdown requires terminal task states, a closed service port and no
remaining marked run processes. SIGINT/SIGTERM execute cleanup; an unconditional
process kill requires the CI fallback or the explicit cleanup command. Investigate
an unverified shutdown before reusing devices. Cleanup is not an environment
rollback. `summary.md` in the runs' parent directory accumulates result links.

## CI and periodic execution

[Workflow](../../.github/workflows/deepseek_agentx_regression.yml) provides:

1. CPU-only tests on pull requests that change the harness.
2. A manual full regression on a self-hosted runner labeled `linux`, `arm64`, `npu`.
3. An optional weekly Monday 02:17 Asia/Shanghai full regression.
4. An `always()` cleanup step, job summary and 30-day evidence upload.

The weekly full run is disabled unless the repository variable
`DEEPSEEK_AGENTX_ENABLE_SCHEDULE=true` is configured. Before enabling it, configure
`DEEPSEEK_AGENTX_CONDA_ENV`, runner environment `PYPTO_DSV4_DSPARK_MODEL_DIR` and
`CANN_ROOT`, queue allocation, disk retention and credentials. Manual dispatch
accepts the environment, model and CANN paths. Scheduling takes effect only when
the workflow is present on the default branch. Adding this code does not install
a local cron job or change repository settings.

Source checkouts, the PTOAS environment and AgentX environment stay on the host;
only evidence is uploaded. Remove old runs only after verifying editable install
references and preserving failure evidence. Each new AgentX version is visible
in the manifest, so workload drift must be reviewed before comparing runs.

## Development validation

```bash
python -m pip install -r scripts/deepseek_agentx/requirements.txt 'pytest<9.2'
python -m pytest tests/regression/deepseek_agentx -q
ruff check scripts/run_deepseek_agentx_regression.py scripts/deepseek_agentx tests/regression/deepseek_agentx
```

The offline suite includes a real local streaming HTTP exchange, wrong-native
identity checks, incomplete/fault evidence, CI parsing and mocked orchestration
through success, queue failure, device failure and SIGTERM cleanup. It does not
allocate devices or alter a service Conda environment. A real 16-device run is
still required to accept a new harness or a changed upstream interface.
