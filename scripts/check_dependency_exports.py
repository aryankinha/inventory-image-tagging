"""Check that both pip dependency lists still describe the committed uv lock."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def package_lines(text):
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


if __name__ == '__main__':
    for filename, groups in [('requirements.txt', ['dev', 'notebook']),
                             ('requirements-phase2.txt', ['dev', 'notebook', 'phase2'])]:
        command = ['uv', 'export', '--locked', '--no-default-groups', '--no-hashes',
                   '--no-emit-project', '--format', 'requirements-txt']
        for group in groups:
            command += ['--group', group]
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        if package_lines((ROOT / filename).read_text()) != package_lines(result.stdout):
            raise SystemExit(f'{filename} differs from uv.lock. Regenerate its uv export.')
        print(filename + ': package versions and platform rules match uv.lock.')
