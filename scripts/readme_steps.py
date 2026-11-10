"""Run a named README command block so CI tests the instructions users read."""
import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read_step(name):
    text = (ROOT / 'README.md').read_text()
    pattern = r'<!-- readme: ' + re.escape(name) + r' -->\s*```bash\n(.*?)\n```'
    blocks = re.findall(pattern, text, flags=re.DOTALL)
    if len(blocks) != 1:
        raise ValueError(f'Expected one README command block named {name!r}.')
    return blocks[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step')
    args = parser.parse_args()
    command = read_step(args.step)
    print(f'README / {args.step}\n{command}', flush=True)
    subprocess.run(['bash', '-euo', 'pipefail', '-c', command], cwd=ROOT, check=True)
