"""One command for each explainable stage of Phase 1."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path,
                        help='Separate data/model/report directory for a fresh reproduction')
    sub = parser.add_subparsers(dest='step', required=True)
    prepare = sub.add_parser('prepare', help='Verify/copy training files and persist the shared split')
    prepare.add_argument('--source', type=Path, help='Optional folder containing the original archives')
    sub.add_parser('eda', help='Audit and visualize training data only')
    sub.add_parser('train', help='Train the eight required model/feature combinations')
    evaluate = sub.add_parser('evaluate', help='Freeze choices, then evaluate the official test set once')
    evaluate.add_argument('--source', type=Path, help='Optional original-archive cache')
    predict = sub.add_parser('predict', help='Predict one 28x28 grayscale benchmark-style image')
    predict.add_argument('image', type=Path)
    args = parser.parse_args()
    if args.work_dir is not None:
        from . import data
        data.ROOT = args.work_dir.resolve()
        data.ROOT.mkdir(parents=True, exist_ok=True)
    if args.step == 'prepare':
        from .data import prepare, load_images, split_indices
        prepare(args.source)
        _, labels = load_images('train')
        a, b = split_indices(labels)
        print(f'Ready: {len(a)} training and {len(b)} validation examples; no test data loaded.')
    elif args.step == 'eda':
        from .eda import run
        run()
    elif args.step == 'train':
        from .train import run
        run()
    elif args.step == 'evaluate':
        from .evaluate import run
        run(args.source)
    else:
        import json
        import joblib
        import numpy as np
        from PIL import Image
        from .data import ROOT, CLASSES, sha256
        from .features import extract
        frozen = json.loads((ROOT / 'reports/frozen.json').read_text())
        selected = next(m for m in frozen['models'] if m['name'] == frozen['selected'])
        with Image.open(args.image) as image:
            if image.size != (28, 28) or image.mode != 'L':
                raise ValueError('Use a 28x28 grayscale image, upright on a dark background; arbitrary-photo accuracy is unmeasured.')
            values = extract(np.asarray(image, dtype=np.uint8)[None], selected['representation'])
        model_path = ROOT / 'models' / f"{selected['name']}.joblib"
        if sha256(model_path) != selected['model_sha256']:
            raise ValueError('Saved model differs from the frozen choice')
        model = joblib.load(model_path)
        print(CLASSES[int(model.predict(values)[0])])


if __name__ == '__main__':
    main()
