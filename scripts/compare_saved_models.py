"""Describe an existing pair of frozen models; never fit models or reopen test images."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from dashboard.analysis import paired_outcomes
from src.inventory_tagging.data import sha256


def compare(work_dir):
    frozen_path = work_dir / 'reports/frozen.json'
    frozen = json.loads(frozen_path.read_text())
    # Both names come from validation records, never from sorting test scores.
    core = [r for r in frozen['models'] if r['family'] != 'rbf_svm']
    baseline = max(core, key=lambda r: (r['macro_f1'], r['accuracy']))['name']
    chosen = frozen['selected']
    path = work_dir / 'reports/test/predictions.npz'
    with np.load(path, allow_pickle=False) as saved:
        report = paired_outcomes(saved['truth'], saved[chosen], saved[baseline])
    return {'first_model': chosen, 'second_model': baseline,
            'frozen_sha256': sha256(frozen_path), 'predictions_sha256': sha256(path), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(compare(args.work_dir.resolve()), indent=2))


if __name__ == '__main__':
    main()
