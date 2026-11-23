#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/local_env.sh"
cd "$ROBOLAB_ROOT"
exec "$ROBOLAB_ROOT/.venv/bin/robolab-dashboard" --output-dir "$ROBOLAB_ROOT/output" --host 127.0.0.1 "$@"
