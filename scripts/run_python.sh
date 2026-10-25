#!/usr/bin/env bash
# Run with the interpreter selected by this clone's setup, without shell activation.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f .runtime-cache/python-path ]]; then
  echo 'First run: bash scripts/setup_mac.sh  (or add --pip)' >&2
  exit 1
fi
runtime="$(cat .runtime-cache/python-path)"
if [[ ! -x "$runtime" ]]; then
  echo 'The Python environment moved. Run setup again in this clone.' >&2
  exit 1
fi
export MPLCONFIGDIR="$PWD/.mpl-cache"
export IPYTHONDIR="$PWD/.runtime-cache/ipython"
export LOKY_MAX_CPU_COUNT=2
exec "$runtime" "$@"
