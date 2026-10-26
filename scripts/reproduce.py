"""Reproduce Phase 1 in a separate folder; never overwrite committed evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'runs/phase1')
    parser.add_argument('--source', type=Path, help='Optional original Fashion-MNIST archive cache')
    args = parser.parse_args()
    target = args.work_dir.resolve()
    if target == ROOT:
        parser.error('Choose a separate work folder. The original project results cannot be overwritten.')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from verify_results import check_results
    summary_path = target / 'reports/test/summary.json'
    if summary_path.exists() and json.loads(summary_path.read_text()).get('status') == 'complete':
        report = check_results(target)
        if report['present_model_hashes_verified'] != report['models_recomputed']:
            raise SystemExit('This completed run is missing model files. Restore them or choose a new work folder; its final test will not be repeated.')
        print('This run is already complete and verified. No training or test evaluation was repeated.')
        print(json.dumps(report, indent=2))
        return
    if (target / 'reports/test/started.json').exists():
        raise SystemExit('A final test was already started here. Keep its evidence and choose a new work folder for another run.')
    (target / 'reports').mkdir(parents=True, exist_ok=True)
    steps = ('evaluate',) if (target / 'reports/frozen.json').exists() else ('prepare', 'eda', 'train', 'evaluate')
    for step in steps:
        command = [sys.executable, '-m', 'src.inventory_tagging.cli', '--work-dir', str(target), step]
        if args.source and step in ('prepare', 'evaluate'):
            command += ['--source', str(args.source.resolve())]
        print(f'Phase 1: {step}', flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps(check_results(target), indent=2))
    print(f'Reproduction completed in {target}')


if __name__ == '__main__':
    main()
