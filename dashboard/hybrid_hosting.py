"""Published seed-42 hybrid for the original experiment, with verified downloads."""
import json
from pathlib import Path

from .hosting import ROOT, MAX_BYTES, download_verified
from src.inventory_tagging.data import sha256

RELEASE_URL = 'https://github.com/aryankinha/inventory-image-tagging/releases/download/phase2-hybrid-v1/'


def published_hybrid(folder):
    folder = Path(folder).resolve()
    if folder != ROOT / 'reports/phase2':
        return None
    path = ROOT / 'assets/hosted_hybrid.json'
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record['frozen_sha256'] != sha256(folder / 'frozen.json'):
        return None
    item = record['hybrids']['42']
    if item['file'] != 'hybrid_seed42.joblib' or type(item['bytes']) is not int or not 0 < item['bytes'] <= MAX_BYTES:
        raise ValueError('Invalid published hybrid record.')
    return record


def ensure_hybrid(folder):
    record = published_hybrid(folder)
    if record is None:
        raise FileNotFoundError('No published hybrid matches this experiment.')
    item = record['hybrids']['42']
    target = Path(folder) / 'checkpoints' / item['file']
    download_verified(target, item, RELEASE_URL + item['file'])
    # The loader verifies this experiment binding and the bytes again.
    path = Path(folder) / 'inference.json'
    if not path.exists():
        path.write_text(json.dumps(record, indent=2) + '\n')
    return record
