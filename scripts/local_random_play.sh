#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$ROBOLAB_ROOT"
exec .venv/bin/python policies/random_play/run.py \
  --headless --device "cuda:${RANDOM_PLAY_GPU:-2}" --num-envs 1 --num-runs 1 \
  --task AnimalsInBinTask --num-steps 300 \
  --kit_args="--/renderer/multiGpu/enabled=false" "$@"
