#!/usr/bin/env bash
# uv cannot put a colon-containing virtualenv path on PATH. A temporary alias fixes that.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mpl-cache}"
export LOKY_MAX_CPU_COUNT=2
export IPYTHONDIR="${IPYTHONDIR:-$PWD/.runtime-cache/ipython}"
export JUPYTER_RUNTIME_DIR="${JUPYTER_RUNTIME_DIR:-$PWD/.runtime-cache/jupyter}"
mkdir -p "$IPYTHONDIR" "$JUPYTER_RUNTIME_DIR"
if [[ "$PWD" == *:* ]]; then
  # `uv sync` itself works before the alias is needed for `uv run`.
  if [[ ! -f .venv/pyvenv.cfg ]]; then
    uv sync --locked
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    key=$(printf '%s' "$PWD" | sha256sum | cut -c 1-12)
  else
    key=$(printf '%s' "$PWD" | shasum -a 256 | cut -c 1-12)
  fi
  alias_path="${TMPDIR:-/tmp}/inventory-uv-$key"
  if [[ ! -e "$alias_path" && ! -L "$alias_path" ]]; then
    ln -s "$PWD/.venv" "$alias_path"
  fi
  if [[ "$(readlink "$alias_path")" != "$PWD/.venv" ]]; then
    echo 'Temporary environment alias belongs to another path; inspect it before continuing.' >&2
    exit 1
  fi
  export UV_PROJECT_ENVIRONMENT="$alias_path"
fi
exec uv "$@"
