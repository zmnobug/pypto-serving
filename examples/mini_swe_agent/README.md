# mini-swe-agent integration

This example connects mini-swe-agent to PyPTO Serving for chat and coding tasks. The Agent can inspect workspace files, run local shell commands, edit code, and run tests. Qwen3-14B inference runs through the Serving engine on an Ascend NPU.

| Transport | Model request path | Use case |
| --- | --- | --- |
| HTTP (default) | Agent -> LiteLLM -> Serving Chat API -> engine | An existing Serving process shared by clients |
| Direct | Agent -> `AsyncLLMEngine.add_request()` | One Agent process starts and owns its engine |

Both paths use the same Serving scheduler, KV cache, and NPU worker implementation. Direct mode does not attach to an already running Serving process. It does not yet adapt Agent decisions to queue or cache state.

## Prerequisites

The commands below use the environments and model installed on this host:

- Repository: `/data/zhaomin/agent/pypto-serving`
- Serving Python: `/data/zhaomin/.conda/envs/zm_pypto/bin/python`
- mini-swe-agent environment: `/data/zhaomin/agent/.venvs/mini-swe-agent`
- Model: `/data/models/Qwen3-14B`
- NPU platform: `a2a3`

Reserve an NPU device before starting either mode. Replace `DEVICE_ID=0` with the assigned device ID. Do not load another model instance on the same device. Initial model loading can take substantially longer than a warm request.

## Direct mode

From the repository root:

```bash
cd /data/zhaomin/agent/pypto-serving
DEVICE_ID=0
bash examples/mini_swe_agent/run_direct.sh \
  --device "$DEVICE_ID" \
  --model-path /data/models/Qwen3-14B \
  --profile chat \
  --task 'Hello'
```

`run_direct.sh` starts the engine with `zm_pypto`, makes the installed mini-swe-agent package available, and sets `PTOAS_ROOT` and `PATH`. No HTTP server is needed. Its default Python and Agent package paths can be overridden with `PYPTO_AGENT_PYTHON` and `PYPTO_AGENT_SITE`.

The `chat` profile suits conversation and file summaries. The default `coding` profile uses mini-swe-agent's coding prompt and supports code repair tasks.

## HTTP mode

Start Serving on the reserved device in one terminal:

```bash
cd /data/zhaomin/agent/pypto-serving
conda activate zm_pypto
DEVICE_ID=0
export PATH=/usr/local/bin/ptoas-bin:$PATH
export PTOAS_ROOT=/usr/local/bin/ptoas-bin
pypto-serving \
  --model /data/models/Qwen3-14B \
  --served-model-name qwen3-agent \
  --backend npu --platform a2a3 --device "$DEVICE_ID" \
  --max-model-len 4096 --max-num-seqs 16 \
  --host 127.0.0.1 --port 8899
```

After the health endpoint responds, start the Agent in another terminal:

```bash
curl --noproxy '*' http://127.0.0.1:8899/health
cd /data/zhaomin/agent/pypto-serving
LITELLM_LOCAL_MODEL_COST_MAP=True \
  /data/zhaomin/agent/.venvs/mini-swe-agent/bin/python \
  examples/mini_swe_agent/interactive.py \
  --transport http --api-base http://127.0.0.1:8899/v1 \
  --profile chat --task 'Hello'
```

The HTTP Agent does not start or stop Serving. Keep its `--context-limit` consistent with the service's actual `--max-model-len`.

## Interaction and coding tasks

At `Execute 1 action(s)?`, press Enter to approve the displayed command. Typing a comment rejects the command and sends the comment back to the Agent. When the Agent proposes completion, press Enter to end the session or type another message to continue.

Commands run inside `--workspace`. Without that option, the Agent creates an empty workspace for the session. For a coding task in an existing project:

```bash
DEVICE_ID=0
bash examples/mini_swe_agent/run_direct.sh \
  --device "$DEVICE_ID" \
  --profile coding \
  --workspace /path/to/project \
  --task 'Read the README, fix the failing tests, and summarize the changes' \
  --verify-command 'python -m unittest -v' \
  --require-change report.py \
  --protect README.md
```

The example verification command and file names must be adjusted for the target project. `--verify-command` runs when the Agent claims completion. A failure rejects completion and returns the result to the Agent. `--require-change` and `--protect` can be repeated to require changed files or protect unchanged files; paths are relative to the workspace. Review each shell command before approving it.

Use repeated `--context-file` options, such as `--context-file README.md --context-file report.py`, to refresh and retain selected files in every model request. Shell commands use the Python interpreter that started the Agent: the separate Agent environment in HTTP mode and `zm_pypto` in direct mode. The startup output shows the workspace and trajectory path; trajectories and verification records are saved under `outputs/mini_swe_agent_interactive/session_*/` by default.

## 16K full-time thinking

Set the Qwen kernel capacity **before starting the engine**:

```bash
export PTO2_QWEN_MAX_SEQ=16384
```

In HTTP mode, also start Serving with `--max-model-len 16384`. In either mode, pass these Agent options:

```text
--context-limit 16384 --max-tokens 8192 --thinking --request-timeout 600
```

The 16K context and 8K output reserve have been verified for a multi-file repair task with thinking enabled throughout on both HTTP and direct mode. Increasing Agent limits alone does not increase the engine's actual capacity. The repair task can finish without actually crossing 4K, so use the capacity probe below to test that boundary separately.

## Reproducible comparison

Reserve an NPU device with `task-submit` or the local device lock before running these commands. Run HTTP and direct cases sequentially on the same device and with the same model and context settings. Each repair run creates a fresh fixture and checks both the public and independent tests:

```bash
export PTO2_QWEN_MAX_SEQ=16384
DEVICE_ID=0
bash examples/mini_swe_agent/run_direct.sh --experiment \
  --device "$DEVICE_ID" --variant thinking8192 \
  --context-limit 16384 --request-timeout 600 --wall-time 1800 \
  --output-dir outputs/direct_repair
```

For HTTP, use `experiment_suite.py` to start and stop a local Serving process and run the same repair variant:

```bash
PTO2_QWEN_MAX_SEQ=16384 /data/zhaomin/.conda/envs/zm_pypto/bin/python \
  examples/mini_swe_agent/experiment_suite.py \
  --device "$DEVICE_ID" --context-limit 16384 \
  --variants thinking8192 --repeats 1 \
  --request-timeout 600 --wall-time 1800 \
  --output-dir outputs/http_repair
```

The direct run writes `summary.json` inside its variant directory; the HTTP suite writes `suite.json`. Compare acceptance, independent tests, calls, tokens, and truncation before comparing task duration. Different action paths can dominate end-to-end time even with the same transport and model settings.

To verify the direct engine's 16K capacity independently of Agent behavior, run:

```bash
bash examples/mini_swe_agent/run_direct.sh --capacity-probe \
  --device "$DEVICE_ID" --context-limit 16384 \
  --output-dir outputs/direct_capacity
```

The probe requests a short prompt, one crossing 4K, and one near the 16K limit, all with thinking enabled. To measure only warm fixed-request latency through each transport on the same device, run:

```bash
bash examples/mini_swe_agent/run_direct.sh --latency-probe \
  --device "$DEVICE_ID" --context-limit 16384 \
  --warmups 2 --samples 10 --order direct,http \
  --output-dir outputs/transport_latency
```

The latency probe loads each mode in turn, records startup separately, and saves per-request times and token counts in `summary.json`. It uses a short, non-thinking response to focus on request-path overhead; it does not measure full Agent task performance.

## Results and checks

- In two fresh 16K, full-thinking repair runs per transport, both HTTP and direct completed one run and timed out on the other. These small samples do not establish a task-success advantage for either path.
- Direct generation passed the thinking-enabled capacity probe with 4,092 and 15,360 prompt tokens. The latter request used 15,558 tokens including output.
- On this host, a fixed 19-input-token, 2-output-token request finished normally in both modes. After two warmups, ten measured requests had median latencies of 98.545 ms direct and 98.552 ms HTTP. This does not show a meaningful speedup from removing the local HTTP hop; it does not predict long-generation or concurrent-load performance.
- HTTP: use `/health` to check the server. For a `task-submit` launch, use `task-submit --status TASK_ID` to check the job.
- Direct: there is no HTTP health endpoint. The engine runs for the lifetime of the Agent process; inspect its terminal output or job status.
- Session output: inspect the printed `trajectory.json` path and any verification records in the same directory.
- The context limit must exceed the output allowance. Incomplete generations are not executed as actions.
- Run the focused CPU tests with `LITELLM_LOCAL_MODEL_COST_MAP=True /data/zhaomin/agent/.venvs/mini-swe-agent/bin/python -m unittest discover -s examples/mini_swe_agent -p 'test_*.py' -q`.

This demo has exercised local shell commands for file reading, code editing, test execution, and task submission. Browser automation, database access, web search, Git operations, and native function calling have not been validated. Direct mode exposes the engine in the Agent process; queue-aware decisions and automatic capacity discovery remain future work.
