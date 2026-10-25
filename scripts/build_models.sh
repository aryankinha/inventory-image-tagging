#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_python.sh scripts/reproduce.py "$@"
