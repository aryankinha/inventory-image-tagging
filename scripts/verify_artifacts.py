"""Check protected reports and samples without loading models or downloading data."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / 'reports/artifact_hashes.json').read_text())
    failures = []
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe path in integrity manifest.')
        path = root / relative
        if not path.is_file():
            failures.append('Missing: ' + name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append('Changed: ' + name)
    return {'status': 'failed' if failures else 'passed', 'files_checked': len(manifest['files']),
            'failures': failures, 'note': 'This manifest protects saved evidence and training examples. It does not replace model checks or verify results of another run.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    report = check(args.root.resolve())
    print(json.dumps(report, indent=2))
    return 1 if report['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
