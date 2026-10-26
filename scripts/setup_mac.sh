#!/usr/bin/env bash
# Prepare this clone without copying another computer's environment.
set -euo pipefail
cd "$(dirname "$0")/.."
project_dir="$PWD"
mkdir -p .runtime-cache/ipython
export MPLCONFIGDIR="$project_dir/.mpl-cache"
export IPYTHONDIR="$project_dir/.runtime-cache/ipython"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$project_dir/.uv-cache}"
method="${1:-uv}"
if [[ "$method" == "--pip" ]]; then
  runner=""
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(not (3,11) <= sys.version_info[:2] < (3,13))'; then
      runner="$candidate"
      break
    fi
  done
  if [[ -z "$runner" ]]; then
    echo 'Install Python 3.12 or 3.11 from python.org, then run this command again.' >&2
    exit 1
  fi
  key=$(printf '%s' "$project_dir" | shasum -a 256 | cut -c 1-12)
  # A venv path containing ':' cannot be used in PATH. Keep pip's environment outside it.
  pip_env="${INVENTORY_PIP_ENV:-$HOME/venvs/inventory-tagging-$key}"
  if [[ "$pip_env" == *:* ]]; then
    echo 'Choose an INVENTORY_PIP_ENV path without a colon.' >&2
    exit 1
  fi
  "$runner" -m venv "$pip_env"
  runtime="$pip_env/bin/python3"
  "$runtime" -m pip install -r requirements.txt
elif [[ "$method" == "uv" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ then retry.' >&2
    echo 'Or use your installed Python 3.11/3.12: bash scripts/setup_mac.sh --pip' >&2
    exit 1
  fi
  uv sync --locked
  runtime="$project_dir/.venv/bin/python3"
else
  echo 'Use: bash scripts/setup_mac.sh [--pip]' >&2
  exit 1
fi
"$runtime" scripts/check_environment.py
"$runtime" scripts/verify_results.py
"$runtime" scripts/verify_artifacts.py
"$runtime" -m ipykernel install --sys-prefix --name inventory-tagging --display-name 'Python (Inventory Tagging)'
printf '%s\n' "$runtime" > .runtime-cache/python-path
printf '\nSetup passed. Open the dashboard with:\n  bash scripts/start_dashboard.sh\n'
echo 'Predictions fetch a verified published model on first use. Training your own models is optional; see README.md.'
