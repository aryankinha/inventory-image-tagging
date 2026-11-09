"""One command per Phase 2 stage. `all` runs them in order."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['ablation', 'select', 'hybrid', 'test', 'all', 'predict', 'export'])
    parser.add_argument('--work-dir', default='.',
                        help='Folder for a fresh run, for example runs/phase2. The saved original run is never overwritten.')
    parser.add_argument('--source', help='Optional folder with the original Fashion-MNIST archives (test stage)')
    parser.add_argument('--image', help='PNG image, or PNG/JPEG with --photo')
    parser.add_argument('--photo', action='store_true', help='Experimental ordinary-photo preparation')
    parser.add_argument('--shadow-background',action='store_true')
    parser.add_argument('--isolate', action='store_true', help='Remove a plain background from one photo item')
    parser.add_argument('--crop', type=float, nargs=4, metavar=('LEFT','TOP','RIGHT','BOTTOM'), help='Crop fractions from 0 to 1; use with --isolate')
    parser.add_argument('--rotation', type=int, choices=[0,90,180,270], default=0)
    parser.add_argument('--invert', action='store_true', help='Invert the prepared photo')
    parser.add_argument('--model', choices=['cnn', 'hybrid'], default='cnn')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.stage in ('predict', 'export'):
        from pathlib import Path
        import json
        from .predict import export_hybrids, load_predictor, predict
        folder = Path(args.work_dir).resolve() / 'reports/phase2'
        if args.stage == 'export':
            print(json.dumps(export_hybrids(folder), indent=2))
            return
        if not args.image:
            parser.error('predict needs --image path/to/picture.png')
        if (args.isolate or args.crop or args.rotation or args.shadow_background) and not args.photo:
            parser.error('Photo preparation options require --photo')
        if (args.crop or args.rotation or args.shadow_background) and not args.isolate:
            parser.error('Crop and rotation require --isolate')
        from dashboard.services import prepare_image
        pixels, _ = prepare_image(Path(args.image).read_bytes(), experimental=args.photo, invert=args.invert, isolate=args.isolate, crop=args.crop, rotation=args.rotation,shadow_background=args.shadow_background)
        print(json.dumps(predict(pixels[None], load_predictor(folder, args.model, args.seed)), indent=2))
        return
    if args.work_dir == '.':
        parser.error('Training stages need a separate --work-dir, for example runs/phase2')
    from . import experiments
    experiments.use_work_dir(args.work_dir)
    stages = {'ablation': experiments.ablation, 'select': experiments.select,
              'hybrid': experiments.hybrid, 'test': lambda: experiments.test(args.source)}
    for name in (stages if args.stage == 'all' else [args.stage]):
        print(f'Phase 2: {name}', flush=True)
        stages[name]()


if __name__ == '__main__':
    try:
        main()
    except ImportError as error:
        raise SystemExit('Phase 2 needs its dependencies. Run: bash scripts/setup_mac.sh --phase2') from error
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f'Phase 2: {error}') from error
