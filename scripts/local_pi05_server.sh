#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
export CUDA_VISIBLE_DEVICES="${PI05_GPU:-1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export OPENPI_DATA_HOME="$ROBOLAB_ROOT/.cache/openpi"
export HF_HOME="$ROBOLAB_ROOT/.cache/huggingface"
cd "$ROBOLAB_ROOT/.local/openpi"
exec .venv/bin/python scripts/serve_policy.py --port "${PI05_PORT:-8010}" \
  policy:checkpoint --policy.config=pi05_droid_jointpos \
  --policy.dir="$ROBOLAB_ROOT/.local/checkpoints/pi05_droid_jointpos" "$@"
