"""Verified CNN and hybrid inference. Images use the existing dashboard preparation rules."""
from io import BytesIO
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..inventory_tagging.data import ROOT, CLASSES, sha256
from . import cnn


def load_predictor(folder=None, kind='cnn', seed=42):
    folder = Path(folder) if folder else ROOT / 'reports/phase2'
    if kind not in ('cnn', 'hybrid'):
        raise ValueError('Choose cnn or hybrid.')
    frozen = json.loads((folder / 'frozen.json').read_text())
    if seed not in frozen['seeds']:
        raise ValueError('Choose a seed recorded in this experiment.')
    if frozen['cnn_config'] not in cnn.CONFIGS:
        raise ValueError('Unknown frozen CNN configuration.')
    torch.set_num_threads(2)
    tag = f"{frozen['cnn_config']}_seed{seed}"
    content = (folder / 'checkpoints' / f'{tag}.pt').read_bytes()
    import hashlib
    if hashlib.sha256(content).hexdigest() != frozen['checkpoint_sha256'][tag]:
        raise ValueError('CNN checkpoint checksum mismatch.')
    model = cnn.SmallCNN(**frozen['cnn_settings']).eval()
    model.load_state_dict(torch.load(BytesIO(content), map_location='cpu', weights_only=True))
    head = None
    if kind == 'hybrid':
        record = json.loads((folder / 'inference.json').read_text())
        if record['frozen_sha256'] != sha256(folder / 'frozen.json'):
            raise ValueError('Hybrid record belongs to another frozen experiment.')
        item = record['hybrids'][str(seed)]
        if Path(item['file']).name != item['file']:
            raise ValueError('Invalid hybrid filename.')
        content = (folder / 'checkpoints' / item['file']).read_bytes()
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('Hybrid classifier checksum mismatch.')
        head = joblib.load(BytesIO(content))
    return model, head


def predict(images, predictor):
    """Return labels and actual latency; model loading and image decoding are excluded."""
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) == 0:
        raise ValueError('Expected uint8 images shaped (N, 28, 28).')
    model, head = predictor
    start = time.perf_counter()
    with torch.inference_mode(), threadpool_limits(limits=2):
        values = cnn.to_tensor(images)
        features = model.embedding(values)
        if head is None:
            probabilities = model.head(features).softmax(1).numpy()
            labels = probabilities.argmax(1)
        else:
            labels = head.predict(features.numpy())
    return {'labels': labels.tolist(), 'categories': [CLASSES[int(i)] for i in labels],
            'elapsed_ms': (time.perf_counter() - start) * 1000,
            'probabilities': probabilities.tolist() if head is None else None}


def export_hybrids(folder=None):
    """Rebuild only the frozen classifier on training features; never choose using test scores."""
    from . import experiments
    folder = Path(folder) if folder else ROOT / 'reports/phase2'
    frozen = json.loads((folder / 'frozen.json').read_text())
    selection = json.loads((folder / 'hybrid_selection.json').read_text())
    if (selection['family'], selection['parameters']) != (frozen['hybrid_family'], frozen['hybrid_parameters']):
        raise ValueError('Hybrid selection differs from frozen choices.')
    if sha256(ROOT / 'data/split.npz') != frozen['split_sha256']:
        raise ValueError('Training split differs from frozen choices.')
    if (folder / 'inference.json').exists():
        existing = json.loads((folder / 'inference.json').read_text())
        if existing['frozen_sha256'] != sha256(folder / 'frozen.json'):
            raise ValueError('Hybrid record belongs to another frozen experiment.')
        complete = all((folder / 'checkpoints' / existing['hybrids'][str(seed)]['file']).is_file() for seed in frozen['seeds'])
        if complete:
            for seed in frozen['seeds']:
                load_predictor(folder, 'hybrid', seed)
            return existing
    x_train, y_train, _, _ = experiments.training_data()
    record = {'frozen_sha256': sha256(folder / 'frozen.json'), 'hybrids': {}}
    torch.set_num_threads(2)
    for seed in frozen['seeds']:
        model, _ = load_predictor(folder, 'cnn', seed)
        features = cnn.forward_all(model, x_train, torch.device('cpu'))[1].numpy()
        head, converged = experiments.fit_classifier(frozen['hybrid_family'], frozen['hybrid_parameters'], features, y_train.numpy())
        if not converged:
            raise ValueError('Frozen hybrid classifier did not converge.')
        filename = f'hybrid_seed{seed}.joblib'
        path = folder / 'checkpoints' / filename
        joblib.dump(head, path, compress=3)
        record['hybrids'][str(seed)] = {'file': filename, 'sha256': sha256(path), 'bytes': path.stat().st_size}
    from ..inventory_tagging.data import write_json
    write_json(folder / 'inference.json', record)
    return record
