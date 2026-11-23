#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$ROBOLAB_ROOT"
exec .venv/bin/python policies/pi0_family/run.py \
  --policy pi05 --headless --device cuda:0 --num-envs 1 --num-runs 1 \
  --task BananaInBowlTask --remote-host 127.0.0.1 --remote-port "${PI05_PORT:-8010}" \
  --kit_args="--/renderer/multiGpu/enabled=false" "$@"
