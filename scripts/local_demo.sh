#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$ROBOLAB_ROOT"
# Replay bundled robot actions; no policy server or model weights are needed.
exec "$ROBOLAB_ROOT/.venv/bin/python" examples/run_recorded.py --headless --device cuda:0 --num_envs 1 \
  --kit_args="--/renderer/multiGpu/enabled=false" "$@"
