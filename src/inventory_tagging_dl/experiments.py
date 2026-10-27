"""Phase 2 stages: ablation, selection, hybrid, and one final test.

Same protocol as Phase 1: 48,000 training and 12,000 validation pictures from the official
training set. Every choice uses validation only. The 10,000 test pictures are used once, after
the choices are written to reports/phase2/frozen.json.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..inventory_tagging.data import ROOT, CLASSES, load_images, prepare, sha256, split_indices, write_json
from ..inventory_tagging.models import build, choices, complexity
from . import cnn

OUT = ROOT / 'reports/phase2'
CHECKPOINTS = ROOT / 'models/phase2'       # every ablation run (not in Git)
FAMILIES = ('logistic_regression', 'linear_svm', 'random_forest')
UPPER_BODY = (0, 2, 4, 6)                  # T-shirt/top, Pullover, Coat, Shirt


def training_data():
    images, labels = load_images('train')
    train_ids, validation_ids = split_indices(labels)
    return (cnn.to_tensor(images[train_ids]), torch.from_numpy(labels[train_ids].astype(np.int64)),
            cnn.to_tensor(images[validation_ids]), torch.from_numpy(labels[validation_ids].astype(np.int64)))


def use_work_dir(path):
    """Write a reproduction under `path`. The committed run (reports/phase2) is never overwritten."""
    global OUT, CHECKPOINTS
    path = Path(path).resolve()
    if path == ROOT:
        raise ValueError('Choose a work folder such as runs/phase2, not the project folder.')
    OUT, CHECKPOINTS = path / 'reports/phase2', path / 'models/phase2'


def load_cnn(config_name, seed, folder=None):
    model = cnn.SmallCNN(**cnn.CONFIGS[config_name])
    model.load_state_dict(torch.load((folder or CHECKPOINTS) / f'{config_name}_seed{seed}.pt', map_location='cpu', weights_only=True))
    return model.eval()


# ---- 1. Ablation -------------------------------------------------------------------------

def ablation():
    """Train every configuration with three seeds. Finished runs are skipped, so it can resume."""
    (OUT / 'ablation').mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    prepare()           # downloads and checks the training archives only if they are missing
    x_train, y_train, x_val, y_val = training_data()
    device = cnn.pick_device()
    print(f'Device: {device}; training {len(x_train)}, validation {len(x_val)}; test files not opened', flush=True)
    for name, config in cnn.CONFIGS.items():
        for seed in cnn.SEEDS:
            tag = f'{name}_seed{seed}'
            if (OUT / 'ablation' / f'{tag}.json').exists():
                continue
            model, history, summary = cnn.train_cnn(config, seed, x_train, y_train, x_val, y_val, device)
            torch.save(model.state_dict(), CHECKPOINTS / f'{tag}.pt')
            history.to_csv(OUT / 'ablation' / f'{tag}_history.csv', index=False)
            write_json(OUT / 'ablation' / f'{tag}.json', {'config': name, 'seed': seed, **config, **summary})
            print(f"{tag}: val macro F1 {summary['validation_macro_f1']:.4f}, best epoch {summary['best_epoch']}/"
                  f"{summary['epochs_run']}, {summary['seconds']:.0f}s", flush=True)
    summarize_ablation()


def summarize_ablation():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / 'ablation').glob('*_seed*.json'))]
    frame = pd.DataFrame(rows)
    metrics = ['validation_macro_f1', 'validation_accuracy', 'validation_loss', 'best_epoch', 'seconds']
    table = frame.groupby('config', sort=False)[metrics].agg(['mean', 'std'])
    table.columns = ['_'.join(c) for c in table.columns]
    table['parameters'] = frame.groupby('config', sort=False).parameters.first()
    table['runs'] = frame.groupby('config', sort=False).size()
    table = table.loc[list(cnn.CONFIGS)].reset_index()
    table.to_csv(OUT / 'ablation_summary.csv', index=False)
    frame.to_csv(OUT / 'ablation_runs.csv', index=False)
    (OUT / 'figures').mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 3.4))
    axis.barh(table.config[::-1], table.validation_macro_f1_mean[::-1], xerr=table.validation_macro_f1_std[::-1],
              color='#4c78a8', capsize=3)
    axis.set_xlim(table.validation_macro_f1_mean.min() - 0.02, table.validation_macro_f1_mean.max() + 0.01)
    axis.set_xlabel('Validation macro F1 (mean and standard deviation of 3 seeds)')
    fig.tight_layout()
    fig.savefig(OUT / 'figures/ablation.png', dpi=160)
    plt.close(fig)
    return table


# ---- 2. Choose the CNN configuration -----------------------------------------------------

def select():
    table = pd.read_csv(OUT / 'ablation_summary.csv')
    if len(table) != len(cnn.CONFIGS) or not (table.runs == len(cnn.SEEDS)).all():
        raise RuntimeError('Run the ablation to completion before selecting.')
    ranked = table.sort_values('validation_macro_f1_mean', ascending=False)
    overall = ranked.iloc[0]
    # The project brief asks for a CNN with dropout, so the final model is the best one that uses it.
    best = ranked[[cnn.CONFIGS[name]['dropout'] > 0 for name in ranked.config]].iloc[0]
    write_json(OUT / 'selection.json', {
        'config': best.config, 'validation_macro_f1_mean': float(best.validation_macro_f1_mean),
        'rule': 'highest mean validation macro F1 over 3 seeds among configurations that use dropout (the brief asks for it)',
        'best_overall_config': overall.config, 'best_overall_validation_macro_f1_mean': float(overall.validation_macro_f1_mean),
        'seed_std_of_selected': float(best.validation_macro_f1_std)})
    print(f'Selected CNN configuration: {best.config} (validation macro F1 {best.validation_macro_f1_mean:.4f}); '
          f'best overall was {overall.config} ({overall.validation_macro_f1_mean:.4f})')


# ---- 3. Hybrid: CNN features + a Phase 1 classifier ---------------------------------------

def classifier(family, parameters):
    model = build(family, parameters)
    # The 64 learned features have no fixed scale, so the linear SVM gets a scaler fitted on training data.
    return make_pipeline(StandardScaler(), model) if family == 'linear_svm' else model


def fit_classifier(family, parameters, features, labels):
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter('always')
        model = classifier(family, parameters).fit(features, labels)
    return model, not any(issubclass(w.category, ConvergenceWarning) for w in caught)


def hybrid():
    config = json.loads((OUT / 'selection.json').read_text())['config']
    x_train, y_train, x_val, y_val = training_data()
    device = cnn.pick_device()
    rows = []
    for seed in cnn.SEEDS:
        model = load_cnn(config, seed)
        features = {'train': cnn.forward_all(model, x_train, device)[1].numpy(),
                    'validation': cnn.forward_all(model, x_val, device)[1].numpy()}
        for family in FAMILIES:
            for parameters in choices(family):
                fitted, converged = fit_classifier(family, parameters, features['train'], y_train.numpy())
                predicted = fitted.predict(features['validation'])
                rows.append({'seed': seed, 'family': family, 'parameters': json.dumps(parameters), 'converged': converged,
                             'validation_accuracy': accuracy_score(y_val.numpy(), predicted),
                             'validation_macro_f1': f1_score(y_val.numpy(), predicted, average='macro')})
                print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'hybrid_validation.csv', index=False)
    grouped = frame.groupby(['family', 'parameters']).agg(f1=('validation_macro_f1', 'mean'), ok=('converged', 'all')).reset_index()
    best = grouped[grouped.ok].sort_values('f1', ascending=False).iloc[0]
    family, parameters = best.family, json.loads(best.parameters)
    # Control: the same classifier on features from an untrained CNN. If it is much worse, the
    # CNN's training (not just its shape) is what makes the features useful.
    control = []
    for seed in cnn.SEEDS:
        torch.manual_seed(seed)
        untrained = cnn.SmallCNN(**cnn.CONFIGS[config]).eval()
        train_features = cnn.forward_all(untrained, x_train, device)[1].numpy()
        fitted, _ = fit_classifier(family, parameters, train_features, y_train.numpy())
        predicted = fitted.predict(cnn.forward_all(untrained, x_val, device)[1].numpy())
        control.append(f1_score(y_val.numpy(), predicted, average='macro'))
    head = [json.loads(p.read_text())['validation_macro_f1'] for p in (OUT / 'ablation').glob(f'{config}_seed*.json')]
    write_json(OUT / 'hybrid_selection.json', {
        'family': family, 'parameters': parameters, 'rule': 'highest mean validation macro F1; all seeds converged',
        'validation_macro_f1_mean': float(best.f1), 'cnn_own_head_validation_macro_f1_mean': float(np.mean(head)),
        'untrained_cnn_features_validation_macro_f1_mean': float(np.mean(control))})
    print(f'Selected hybrid classifier: {family} {parameters} (validation macro F1 {best.f1:.4f}; '
          f'untrained-features control {np.mean(control):.4f})')


# ---- 4. One final test -------------------------------------------------------------------

def time_predictor(predict, images):
    """Median single-picture latency and batch throughput, timed like Phase 1 (uint8 pictures to labels)."""
    result = {}
    for batch_size, count, repeats in ((1, 1, 20), (128, 1024, 5)):
        sample = images[:count]

        def run():
            for first in range(0, count, batch_size):
                predict(sample[first:first + batch_size])
        run()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            run()
            times.append((time.perf_counter() - start) * 1000 / count)
        result[batch_size] = times
    return {'single_ms_median': float(np.median(result[1])), 'single_ms_p95': float(np.quantile(result[1], .95)),
            'batch_ms_per_image': float(np.median(result[128])),
            'batch_images_per_second': float(1000 / np.median(result[128]))}


def paired_bootstrap(correct_a, correct_b, resamples=2000):
    """95% interval for accuracy(A) - accuracy(B) on the same test pictures."""
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(correct_a), (resamples, len(correct_a)))
    differences = correct_a[draws].mean(1) - correct_b[draws].mean(1)
    return {'difference': float(correct_a.mean() - correct_b.mean()),
            'low': float(np.quantile(differences, .025)), 'high': float(np.quantile(differences, .975))}


def test(source=None):
    if (OUT / 'test/results.csv').exists():
        raise FileExistsError('Final test already finished; use the saved outputs instead of tuning on them.')
    config = json.loads((OUT / 'selection.json').read_text())['config']
    chosen = json.loads((OUT / 'hybrid_selection.json').read_text())
    family, parameters = chosen['family'], chosen['parameters']
    phase1 = json.loads((ROOT / 'reports/frozen.json').read_text())['selected']
    frozen = {'cnn_config': config, 'cnn_settings': cnn.CONFIGS[config], 'training': cnn.TRAIN, 'seeds': list(cnn.SEEDS),
              'hybrid_family': family, 'hybrid_parameters': parameters, 'phase1_selected': phase1,
              'checkpoint_sha256': {f'{config}_seed{s}': sha256(CHECKPOINTS / f'{config}_seed{s}.pt') for s in cnn.SEEDS},
              'split_sha256': sha256(ROOT / 'data/split.npz'), 'phase1_frozen_sha256': sha256(ROOT / 'reports/frozen.json')}
    started = OUT / 'test/started.json'
    if started.exists():
        # A run that crashed before writing results may be retried, but only with identical choices.
        if json.loads((OUT / 'frozen.json').read_text()) != frozen:
            raise FileExistsError('Choices changed after the test started; start a new Phase 2 folder.')
    else:
        write_json(OUT / 'frozen.json', frozen)
        write_json(started, {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                             'frozen_sha256': sha256(OUT / 'frozen.json')})
    prepare(source, include_test=True)
    images, truth = load_images('test')
    x_test = cnn.to_tensor(images)
    x_train, y_train, _, _ = training_data()
    old = np.load(ROOT / 'reports/test/predictions.npz')
    assert np.array_equal(old['truth'], truth), 'Phase 1 and Phase 2 must score the same test pictures'
    device = cnn.pick_device()

    predictions, rows, fitted = {'truth': truth}, [], {}
    for seed in cnn.SEEDS:
        model = load_cnn(config, seed)
        logits, test_features = cnn.forward_all(model, x_test, device)
        train_features = cnn.forward_all(model, x_train, device)[1].numpy()
        classical, _ = fit_classifier(family, parameters, train_features, y_train.numpy())
        fitted[seed] = (model, classical)
        predictions[f'cnn_seed{seed}'] = logits.argmax(1).numpy()
        predictions[f'hybrid_seed{seed}'] = classical.predict(test_features.numpy())
        for kind in ('cnn', 'hybrid'):
            predicted = predictions[f'{kind}_seed{seed}']
            rows.append({'model': kind, 'seed': seed, 'test_accuracy': accuracy_score(truth, predicted),
                         'test_macro_f1': f1_score(truth, predicted, average='macro')})
    np.savez_compressed(OUT / 'test/predictions.npz', **predictions)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'test/results.csv', index=False)

    # Costs, measured on the CPU with two threads like Phase 1 (seed 42).
    torch.set_num_threads(2)
    model, classical = fitted[42]
    model = model.cpu().eval()      # the scoring above may have left it on the GPU
    with torch.inference_mode(), threadpool_limits(limits=2):
        cnn_cost = time_predictor(lambda b: model(cnn.to_tensor(b)).argmax(1), images)
        hybrid_cost = time_predictor(lambda b: classical.predict(model.embedding(cnn.to_tensor(b)).numpy()), images)
    cnn_file = CHECKPOINTS / f'{config}_seed42.pt'
    costs = {'cnn': {**cnn_cost, 'parameters': cnn.count_parameters(model), 'file_bytes': cnn_file.stat().st_size},
             'hybrid': {**hybrid_cost, 'cnn_parameters_without_head': cnn.count_parameters(model, head=False),
                        'classifier': complexity(classical)}}

    correct = {k: v == truth for k, v in predictions.items() if k != 'truth'}
    correct['phase1'] = old[phase1] == truth
    scores = {}
    for kind in ('cnn', 'hybrid'):
        for metric in ('test_accuracy', 'test_macro_f1'):
            values = results[results.model == kind][metric]
            scores[f'{kind}_{metric}_mean'], scores[f'{kind}_{metric}_std'] = float(values.mean()), float(values.std())
    summary = {
        'phase1_selected': phase1, 'phase1_test_accuracy': float(correct['phase1'].mean()),
        'phase1_test_macro_f1': float(f1_score(truth, old[phase1], average='macro')), **scores,
        'cnn_vs_phase1_seed42': paired_bootstrap(correct['cnn_seed42'], correct['phase1']),
        'hybrid_vs_cnn_seed42': paired_bootstrap(correct['hybrid_seed42'], correct['cnn_seed42']),
        'costs': costs, 'test_images': int(len(truth))}
    write_json(OUT / 'test/summary.json', summary)

    # Tables for the error analysis.
    recalls = pd.DataFrame({'category': CLASSES,
                            'phase1_' + phase1: recall_score(truth, old[phase1], average=None),
                            'cnn_mean': np.mean([recall_score(truth, predictions[f'cnn_seed{s}'], average=None) for s in cnn.SEEDS], 0),
                            'hybrid_mean': np.mean([recall_score(truth, predictions[f'hybrid_seed{s}'], average=None) for s in cnn.SEEDS], 0)})
    recalls.to_csv(OUT / 'test/per_class_recall.csv', index=False)
    matrices = {'phase1': confusion_matrix(truth, old[phase1]), 'cnn': confusion_matrix(truth, predictions['cnn_seed42']),
                'hybrid': confusion_matrix(truth, predictions['hybrid_seed42'])}
    pd.DataFrame(matrices['cnn'], index=CLASSES, columns=CLASSES).to_csv(OUT / 'test/confusion_cnn_seed42.csv')
    pairs = [{'true': CLASSES[a], 'predicted': CLASSES[b], **{k: int(m[a, b]) for k, m in matrices.items()}}
             for a in UPPER_BODY for b in UPPER_BODY if a != b]
    pd.DataFrame(pairs).to_csv(OUT / 'test/upper_body_confusions.csv', index=False)
    comparison = pd.read_csv(ROOT / 'reports/test/results.csv')[['name', 'test_accuracy', 'test_macro_f1']]
    comparison = pd.concat([comparison, pd.DataFrame([
        {'name': 'cnn (mean of 3 seeds)', 'test_accuracy': summary['cnn_test_accuracy_mean'], 'test_macro_f1': summary['cnn_test_macro_f1_mean']},
        {'name': f'hybrid cnn + {family} (mean of 3 seeds)', 'test_accuracy': summary['hybrid_test_accuracy_mean'],
         'test_macro_f1': summary['hybrid_test_macro_f1_mean']}])], ignore_index=True)
    comparison.to_csv(OUT / 'test/comparison.csv', index=False)

    (OUT / 'checkpoints').mkdir(exist_ok=True)
    for seed in cnn.SEEDS:      # small enough to keep in Git so the notebook runs without retraining
        (OUT / 'checkpoints' / f'{config}_seed{seed}.pt').write_bytes((CHECKPOINTS / f'{config}_seed{seed}.pt').read_bytes())
    figures(comparison, recalls, matrices['cnn'], images, truth, predictions, old[phase1], config)
    print(json.dumps({k: v for k, v in summary.items() if k != 'costs'}, indent=2))


# ---- Figures -----------------------------------------------------------------------------

def figures(comparison, recalls, confusion, images, truth, predictions, phase1_predicted, config):
    folder = OUT / 'figures'
    folder.mkdir(exist_ok=True)
    history = pd.read_csv(OUT / f'ablation/{config}_seed42_history.csv')
    fig, axis = plt.subplots(figsize=(5.5, 3.2))
    axis.plot(history.epoch, history.train_loss, label='training loss')
    axis.plot(history.epoch, history.validation_loss, label='validation loss')
    best = history.validation_loss.idxmin()
    axis.axvline(history.epoch[best], color='gray', linestyle=':', label='epoch kept (early stopping)')
    axis.set_xlabel('epoch')
    axis.legend()
    fig.tight_layout()
    fig.savefig(folder / 'learning_curves.png', dpi=160)
    plt.close(fig)

    ordered = comparison.sort_values('test_accuracy')
    colors = ['#e45756' if n.startswith(('cnn', 'hybrid')) else '#9ecae9' for n in ordered.name]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.barh(ordered.name, ordered.test_accuracy * 100, color=colors)
    axis.set_xlim(80, ordered.test_accuracy.max() * 100 + 1)
    axis.set_xlabel('Test accuracy (%)')
    fig.tight_layout()
    fig.savefig(folder / 'test_comparison.png', dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 3.4))
    positions = np.arange(10)
    for offset, column, label in ((-0.27, recalls.columns[1], 'Phase 1 best'), (0, 'cnn_mean', 'CNN'), (0.27, 'hybrid_mean', 'Hybrid')):
        axis.bar(positions + offset, recalls[column] * 100, 0.27, label=label)
    axis.set_xticks(positions, recalls.category, rotation=40, ha='right')
    axis.set_ylabel('Recall (%)')
    axis.set_ylim(40, 100)
    axis.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(folder / 'recall_comparison.png', dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    normalized = confusion / confusion.sum(1, keepdims=True)
    axis.imshow(normalized, cmap='Blues')
    axis.set_xticks(positions, CLASSES, rotation=60, ha='right')
    axis.set_yticks(positions, CLASSES)
    axis.set_xlabel('predicted')
    axis.set_ylabel('actual')
    for a in range(10):
        for b in range(10):
            axis.text(b, a, f'{normalized[a, b]:.2f}', ha='center', va='center', fontsize=6,
                      color='white' if normalized[a, b] > .5 else 'black')
    fig.tight_layout()
    fig.savefig(folder / 'confusion_cnn.png', dpi=160)
    plt.close(fig)

    # Examples: upper-body pictures the CNN gets right and Phase 1 gets wrong, and ones both get wrong.
    rng = np.random.default_rng(0)
    cnn_predicted = predictions['cnn_seed42']
    upper = np.isin(truth, UPPER_BODY)
    groups = (('CNN right, Phase 1 wrong', upper & (cnn_predicted == truth) & (phase1_predicted != truth)),
              ('Both wrong', upper & (cnn_predicted != truth) & (phase1_predicted != truth)))
    fig, axes = plt.subplots(2, 6, figsize=(9, 3.9))
    for row, (title, mask) in enumerate(groups):
        chosen = rng.choice(np.flatnonzero(mask), 6, replace=False)
        for axis, index in zip(axes[row], chosen):
            axis.imshow(images[index], cmap='gray')
            axis.set_title(f'actual: {CLASSES[truth[index]]}\nPhase 1: {CLASSES[phase1_predicted[index]]}\nCNN: {CLASSES[cnn_predicted[index]]}', fontsize=6)
            axis.axis('off')
        axes[row][0].text(-0.15, 0.5, title, transform=axes[row][0].transAxes, rotation=90, va='center', ha='right', fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / 'error_examples.png', dpi=160)
    plt.close(fig)
