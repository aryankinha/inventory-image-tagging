#!/usr/bin/env bash
# Retrain the Phase 2 CNN study from scratch into runs/phase2 (about 45 minutes on an Apple M-series Mac).
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_python.sh -m src.inventory_tagging_dl.cli all --work-dir "${1:-runs/phase2}"
