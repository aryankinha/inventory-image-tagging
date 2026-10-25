#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_python.sh -m streamlit run "$PWD/app.py" "$@"
