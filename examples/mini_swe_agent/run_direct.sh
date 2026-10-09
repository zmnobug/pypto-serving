#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
serving_python="${PYPTO_AGENT_PYTHON:-/data/zhaomin/.conda/envs/zm_pypto/bin/python}"
agent_site="${PYPTO_AGENT_SITE:-/data/zhaomin/agent/.venvs/mini-swe-agent/lib/python3.10/site-packages}"
serving_site="$("$serving_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

export PYTHONPATH="$serving_site:$agent_site:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/usr/local/bin/ptoas-bin:$PATH"
export PTOAS_ROOT="/usr/local/bin/ptoas-bin"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1

exec "$serving_python" "$script_dir/interactive.py" --transport direct "$@"
