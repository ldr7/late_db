#!/usr/bin/env sh
if [ -n "${ZSH_VERSION:-}" ]; then
    LATE_DB_ENV_PATH="${(%):-%N}"
elif [ -n "${BASH_SOURCE:-}" ]; then
    LATE_DB_ENV_PATH="$BASH_SOURCE"
else
    LATE_DB_ENV_PATH="$0"
fi
LATE_DB_ROOT="$(CDPATH= cd -- "$(dirname -- "$LATE_DB_ENV_PATH")" && pwd)"
unset LATE_DB_ENV_PATH
export UV_CACHE_DIR="$LATE_DB_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$LATE_DB_ROOT/.uv-python"
export HF_HOME="$LATE_DB_ROOT/data/universal_retrieval/.hf_home"
if [ -x "$LATE_DB_ROOT/.cargo/bin/cargo" ]; then
    export CARGO_HOME="$LATE_DB_ROOT/.cargo"
    export PATH="$CARGO_HOME/bin:$PATH"
fi
if [ -d "$LATE_DB_ROOT/.rustup/toolchains" ]; then
    export RUSTUP_HOME="$LATE_DB_ROOT/.rustup"
fi
if [ -d "$LATE_DB_ROOT/.venv" ]; then
    export VIRTUAL_ENV="$LATE_DB_ROOT/.venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
fi
