"""Read original Fashion-MNIST files and save one shared, stratified split.

IDX is a short binary header followed by unsigned-byte pixels or labels.
The loader checks the publisher's checksums before reading those bytes.
"""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import struct
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
SOURCE = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
CHECKSUMS = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}
SEED = 42


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(source_dir=None, include_test=False):
    """Copy a verified local cache or download missing original files.

    Training is the default. Test archives are requested only by final evaluation.
    An already-present, damaged file is reported instead of silently replaced.
    """
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    for name, expected in CHECKSUMS.items():
        if name.startswith('t10k') and not include_test:
            continue
        target = raw / name
        if not target.exists():
            temporary = target.with_suffix('.part')
            try:
                if source_dir is not None:
                    shutil.copyfile(Path(source_dir) / name, temporary)
                else:
                    with urllib.request.urlopen(SOURCE + name, timeout=120) as response:
                        with temporary.open('wb') as output:
                            shutil.copyfileobj(response, output)
                if hashlib.md5(temporary.read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Invalid downloaded archive: {name}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if hashlib.md5(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checksum failed: {target}')


def load_images(split='train'):
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    prefix = 'train' if split == 'train' else 't10k'
    count = 60000 if split == 'train' else 10000
    paths = [ROOT / 'data/raw' / f'{prefix}-{kind}-idx{dim}-ubyte.gz'
             for kind, dim in [('images', 3), ('labels', 1)]]
    for path in paths:
        if hashlib.md5(path.read_bytes()).hexdigest() != CHECKSUMS[path.name]:
            raise ValueError(f'Checksum failed: {path}')
    with gzip.open(paths[0], 'rb') as stream:
        if struct.unpack('>IIII', stream.read(16)) != (2051, count, 28, 28):
            raise ValueError('Invalid image header')
        images = np.frombuffer(stream.read(), dtype=np.uint8).reshape(count, 28, 28).copy()
    with gzip.open(paths[1], 'rb') as stream:
        if struct.unpack('>II', stream.read(8)) != (2049, count):
            raise ValueError('Invalid label header')
        labels = np.frombuffer(stream.read(), dtype=np.uint8).copy()
    if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, count // 10)):
        raise ValueError('Unexpected class counts')
    return images, labels


def split_indices(labels):
    """48k training, 12k validation. IDs refer to original training images."""
    train, val = train_test_split(np.arange(len(labels)), test_size=0.2,
                                 random_state=SEED, stratify=labels)
    path = ROOT / 'data/split.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if not (np.array_equal(train, saved['train']) and np.array_equal(val, saved['validation'])):
                raise ValueError('Saved split differs from the declared protocol')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, train=train, validation=val)
    return train, val
