"""Start the README dashboard command, check HTTP health, then stop our server."""
import argparse
import shlex
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

from readme_steps import ROOT, read_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', choices=['dashboard', 'manual-dashboard'], default='dashboard')
    args = parser.parse_args()
    log = ROOT / '.runtime-cache/readme-dashboard.log'
    log.parent.mkdir(exist_ok=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    command = shlex.split(read_step(args.step))
    command += ['--server.port', str(port), '--server.address', '127.0.0.1', '--server.headless', 'true']
    with log.open('w') as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Dashboard stopped during startup: ' + log.read_text())
                try:
                    with urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=2) as response:
                        healthy = response.status == 200 and response.read().strip() == b'ok'
                    if healthy:
                        with urlopen(f'http://127.0.0.1:{port}/', timeout=2) as response:
                            if response.status == 200:
                                print('README dashboard command: HTTP page and health checks passed.')
                                break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError('Dashboard did not become ready: ' + log.read_text())
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
