"""One command per Phase 2 stage. `all` runs them in order."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['ablation', 'select', 'hybrid', 'test', 'all'])
    parser.add_argument('--work-dir', required=True,
                        help='Folder for a fresh run, for example runs/phase2. The saved original run is never overwritten.')
    parser.add_argument('--source', help='Optional folder with the original Fashion-MNIST archives (test stage)')
    args = parser.parse_args()
    from . import experiments
    experiments.use_work_dir(args.work_dir)
    stages = {'ablation': experiments.ablation, 'select': experiments.select,
              'hybrid': experiments.hybrid, 'test': lambda: experiments.test(args.source)}
    for name in (stages if args.stage == 'all' else [args.stage]):
        print(f'Phase 2: {name}', flush=True)
        stages[name]()


if __name__ == '__main__':
    main()
