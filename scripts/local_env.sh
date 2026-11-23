#!/usr/bin/env bash
# Source this file to use the isolated environment installed in this checkout.
ROBOLAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ROBOLAB_ROOT
export PATH="$ROBOLAB_ROOT/.local/bin:$ROBOLAB_ROOT/.venv/bin:$PATH"
export UV_CACHE_DIR="$ROBOLAB_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROBOLAB_ROOT/.cache/python"
export XDG_CACHE_HOME="$ROBOLAB_ROOT/.cache"
export XDG_CONFIG_HOME="$ROBOLAB_ROOT/.runtime/config"
export OMNI_KIT_ACCEPT_EULA=Y
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$ROBOLAB_ROOT/output" "$ROBOLAB_ROOT/.runtime/logs"
