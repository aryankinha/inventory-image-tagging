"""Fetch published Phase 1 weights on demand, checking bytes before loading them."""
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RELEASE_URL = 'https://github.com/aryankinha/inventory-image-tagging/releases/download/phase1-models-v1/'
MAX_BYTES = 120_000_000


def published_model(work_dir, record):
    # Reproductions must use their own weights, never original-run downloads.
    if Path(work_dir).resolve() != ROOT:
        return None
    path = ROOT / 'assets/hosted_models.json'
    if not path.is_file():
        return None
    models = json.loads(path.read_text())['models']
    entry = models.get(record['name'])
    if entry is None or entry['sha256'] != record['model_sha256']:
        return None
    name = record['name']
    if Path(name).name != name or '/' in name or '\\' in name or name in ('.', '..'):
        raise ValueError('Invalid published model name.')
    if not isinstance(entry['bytes'], int) or not 0 < entry['bytes'] <= MAX_BYTES:
        raise ValueError('Invalid published model size.')
    return entry


def model_available(work_dir, record):
    return (Path(work_dir) / 'models' / (record['name'] + '.joblib')).is_file() or published_model(work_dir, record) is not None


def ensure_model(work_dir, record):
    target = Path(work_dir) / 'models' / (record['name'] + '.joblib')
    # Existing files are still checked by the original verified loader.
    if target.is_file():
        return target
    entry = published_model(work_dir, record)
    if entry is None:
        raise FileNotFoundError('The saved model is unavailable for this results folder.')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.download', delete=False) as output:
            temporary = Path(output.name)
            digest, count = hashlib.sha256(), 0
            with urlopen(RELEASE_URL + target.name, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    count += len(chunk)
                    if count > entry['bytes']:
                        raise ValueError('Downloaded model is larger than its published size.')
                    digest.update(chunk)
                    output.write(chunk)
            if count != entry['bytes'] or digest.hexdigest() != record['model_sha256']:
                raise ValueError('Downloaded model checksum mismatch. Please try again later.')
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
