"""Reproduce in a new folder; never overwrite committed final evidence."""
import argparse
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work-dir', type=Path, default=root / 'runs/reproduction')
parser.add_argument('--source', type=Path, help='Optional original Fashion-MNIST archive cache')
args = parser.parse_args()
for step in ('prepare', 'eda', 'train', 'improve', 'evaluate'):
    command = [sys.executable, '-m', 'src.inventory_tagging.cli',
               '--work-dir', str(args.work_dir.resolve()), step]
    if args.source and step in ('prepare', 'evaluate'):
        command += ['--source', str(args.source.resolve())]
    print(f'Phase 1: {step}', flush=True)
    subprocess.run(command, cwd=root, check=True)
print(f'Reproduction completed in {args.work_dir.resolve()}')
