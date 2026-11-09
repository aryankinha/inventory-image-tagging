"""Check Phase 2 records, predictions and verified image inference without training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from src.inventory_tagging.data import sha256


def check(folder=None, replay=False):
    folder = Path(folder) if folder else ROOT / 'reports/phase2'
    read = lambda name: json.loads((folder / name).read_text())
    frozen, selection, hybrid = read('frozen.json'), read('selection.json'), read('hybrid_selection.json')
    if (folder/'artifact_hashes.json').exists():
        for name, expected in read('artifact_hashes.json')['files'].items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts or sha256(ROOT/relative) != expected:
                raise ValueError('Phase 2 protected file changed: '+name)
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    require(sha256(folder / 'frozen.json') == read('test/started.json')['frozen_sha256'], 'Frozen choices changed.')
    require(sha256(ROOT / 'reports/frozen.json') == frozen['phase1_frozen_sha256'], 'Phase 1 choices changed.')
    require(selection['config'] == frozen['cnn_config'], 'CNN selection differs.')
    require((hybrid['family'], hybrid['parameters']) == (frozen['hybrid_family'], frozen['hybrid_parameters']), 'Hybrid selection differs.')
    table = pd.read_csv(folder / 'ablation_summary.csv')
    runs = pd.read_csv(folder / 'ablation_runs.csv')
    require(len(runs) == len(table) * len(frozen['seeds']) == 21, 'Incomplete ablation study.')
    require(not runs.duplicated(['config', 'seed']).any(), 'Duplicate ablation runs.')
    for row in runs.itertuples():
        history = pd.read_csv(folder / f'ablation/{row.config}_seed{row.seed}_history.csv')
        best = history.loc[history.validation_loss.idxmin()]
        require(int(best.epoch) == row.best_epoch and len(history) == row.epochs_run, 'Best epoch/history mismatch.')
        require(np.isclose(best.validation_macro_f1, row.validation_macro_f1), 'Best epoch score mismatch.')
    for row in table.itertuples():
        group = runs[runs.config == row.config]
        require(set(group.seed) == set(frozen['seeds']), 'Missing ablation seed.')
        require(np.isclose(group.validation_macro_f1.mean(), row.validation_macro_f1_mean), 'Ablation mean mismatch.')
        require(np.isclose(group.validation_macro_f1.std(), row.validation_macro_f1_std), 'Ablation spread mismatch.')
    allowed = set(runs[runs.dropout > 0].config)
    require(table[table.config.isin(allowed)].sort_values('validation_macro_f1_mean').iloc[-1].config == selection['config'], 'Selection rule mismatch.')
    hv = pd.read_csv(folder / 'hybrid_validation.csv')
    ranking = hv.groupby(['family', 'parameters']).agg(f1=('validation_macro_f1','mean'), ok=('converged','all')).reset_index()
    best = ranking[ranking.ok].sort_values('f1').iloc[-1]
    require(best.family == hybrid['family'] and json.loads(best.parameters) == hybrid['parameters'], 'Hybrid ranking mismatch.')
    results = pd.read_csv(folder / 'test/results.csv')
    summary = read('test/summary.json')
    recalls = pd.read_csv(folder / 'test/per_class_recall.csv')
    with np.load(folder / 'test/predictions.npz', allow_pickle=False) as predictions, np.load(ROOT / 'reports/test/predictions.npz', allow_pickle=False) as phase1:
        truth = predictions['truth']
        require(np.array_equal(truth, phase1['truth']) and np.array_equal(np.bincount(truth), [1000]*10), 'Test images differ.')
        require(set(zip(results.model, results.seed)) == {(k,s) for k in ('cnn','hybrid') for s in frozen['seeds']}, 'Incomplete test results.')
        for row in results.itertuples():
            guessed = predictions[f'{row.model}_seed{row.seed}']
            require(np.isclose(accuracy_score(truth, guessed), row.test_accuracy), 'Accuracy mismatch.')
            require(np.isclose(f1_score(truth, guessed, average='macro'), row.test_macro_f1), 'Macro F1 mismatch.')
        for kind in ('cnn','hybrid'):
            group = results[results.model == kind]
            for metric in ('test_accuracy','test_macro_f1'):
                require(np.isclose(group[metric].mean(), summary[f'{kind}_{metric}_mean']), 'Test mean mismatch.')
                require(np.isclose(group[metric].std(), summary[f'{kind}_{metric}_std']), 'Test spread mismatch.')
            expected = np.mean([recall_score(truth,predictions[f'{kind}_seed{s}'],average=None) for s in frozen['seeds']],0)
            require(np.allclose(expected,recalls[f'{kind}_mean']), 'Per-class recall mismatch.')
        matrices = {kind:confusion_matrix(truth,predictions[f'{kind}_seed42']) for kind in ('cnn','hybrid')}
        matrices['phase1'] = confusion_matrix(truth,phase1[frozen['phase1_selected']])
        require(np.array_equal(matrices['cnn'],pd.read_csv(folder/'test/confusion_cnn_seed42.csv',index_col=0)), 'Confusion matrix mismatch.')
        from src.inventory_tagging_dl.experiments import paired_bootstrap
        for key, a, b in [('cnn_vs_phase1_seed42', predictions['cnn_seed42'], phase1[frozen['phase1_selected']]), ('hybrid_vs_cnn_seed42', predictions['hybrid_seed42'], predictions['cnn_seed42'])]:
            actual=paired_bootstrap(a==truth,b==truth)
            require(all(np.isclose(actual[k],summary[key][k]) for k in actual), 'Paired bootstrap mismatch.')
        from src.inventory_tagging.data import CLASSES
        for row in pd.read_csv(folder/'test/upper_body_confusions.csv').itertuples():
            a,b=CLASSES.index(row.true),CLASSES.index(row.predicted)
            require(all(matrices[k][a,b] == getattr(row,k) for k in matrices), 'Upper-body count mismatch.')
    for tag, expected in frozen['checkpoint_sha256'].items():
        require(sha256(folder/'checkpoints'/f'{tag}.pt') == expected, 'Checkpoint checksum mismatch.')
    from src.inventory_tagging_dl.predict import load_predictor, predict
    from PIL import Image
    image = np.asarray(Image.open(ROOT/'assets/sample.png'))[None]
    demos = {}
    for kind in ('cnn','hybrid'):
        if kind == 'hybrid':
            if not (folder/'inference.json').exists():
                continue
            record=read('inference.json')
            if not all((folder/'checkpoints'/r['file']).is_file() for r in record['hybrids'].values()):
                continue
        demos[kind] = predict(image,load_predictor(folder,kind))
    replay_rows = []
    if replay:
        from src.inventory_tagging.data import load_images
        from src.inventory_tagging_dl import cnn
        import torch
        torch.set_num_threads(2)
        images, labels = load_images('test')
        with np.load(folder/'test/predictions.npz',allow_pickle=False) as saved:
            for seed in frozen['seeds']:
                for kind in demos:
                    predictor=load_predictor(folder,kind,seed)
                    guessed=[]
                    for start in range(0,len(images),128):
                        guessed.extend(predict(images[start:start+128],predictor)['labels'])
                    agreement=float(np.mean(np.asarray(guessed)==saved[f'{kind}_seed{seed}']))
                    require(agreement > .999, 'Checkpoint replay differs from saved predictions.')
                    replay_rows.append({'model':kind,'seed':seed,'prediction_agreement':agreement,'accuracy':float(np.mean(np.asarray(guessed)==labels))})
    return {'status':'passed','hybrid_ready': 'hybrid' in demos,'ablation_runs':len(runs),'test_scores_recomputed':len(results),'checkpoint_hashes_verified':len(frozen['seeds']),'sample_predictions':demos,'test_replay':replay_rows}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path)
    parser.add_argument('--replay',action='store_true',help='Replay saved models on local official test archives; no fitting or selection')
    args=parser.parse_args()
    print(json.dumps(check(args.folder,args.replay),indent=2))
