"""Generate narrative and LaTeX result tables from measured evidence, never fixed scores."""
from pathlib import Path
import json
import shutil
import pandas as pd

root = Path(__file__).resolve().parents[1]
results = pd.read_csv(root / 'reports/test/results.csv')
validation = pd.read_csv(root / 'reports/validation_results.csv')
frozen = json.loads((root / 'reports/frozen.json').read_text())
selected = results.set_index('name').loc[frozen['selected']]
per_class = pd.read_csv(root / 'reports/test/per_class.csv')
recall = per_class[per_class.name == frozen['selected']]
confusions = pd.read_csv(root / 'reports/test/upper_body_confusions.csv')


def label(name):
    return name.replace('hog_fine', 'Fine HOG').replace('hog', 'HOG').replace('pixels', 'Pixels').replace('_logistic_regression', ' + LR').replace('_random_forest', ' + RF').replace('_linear_svm', ' + linear SVM').replace('_rbf_svm', ' + RBF SVM').replace('_knn', ' + k-NN')


core = results[results.family != 'rbf_svm']
core_best = core.sort_values('validation_macro_f1', ascending=False).iloc[0]
lines = ['# Phase 1 findings', '',
         f"We selected {label(frozen['selected'])} using validation macro F1 before evaluating the official test set. Its test accuracy is {selected.test_accuracy:.2%}, with macro F1 {selected.test_macro_f1:.4f}.", '',
         'Every model trains on the same 48,000 images. We choose settings using 12,000 validation images, then evaluate on the official 10,000 test images. We test two settings for each of eight core combinations and three additional RBF combinations.', '',
         '## Test results', '', '| Model | Test accuracy | Macro F1 |', '|---|---:|---:|']
for row in results.itertuples():
    lines.append(f'| {label(row.name)} | {row.test_accuracy:.2%} | {row.test_macro_f1:.4f} |')
lines += ['', '## What HOG changes in the core experiment', '']
for family in ('logistic_regression','knn','random_forest','linear_svm'):
    pairs = core[core.family == family].set_index('representation')
    delta = pairs.loc['hog','test_macro_f1'] - pairs.loc['pixels','test_macro_f1']
    lines.append(f'- {family.replace("_", " ")}: HOG minus pixels in macro F1 = {delta:+.4f}.')
lines += ['', 'Each pair uses the same images and classifier family. We tune the settings separately for pixels and HOG, so the difference includes that tuning. HOG does not help every classifier.', '',
          '## Core models and RBF experiments', '',
          f"Validation selects {label(core_best['name'])} among the eight core models. Its test accuracy is {core_best.test_accuracy:.2%}. Across all experiments, validation selects {label(frozen['selected'])}, with test accuracy {selected.test_accuracy:.2%}. We fixed both choices before examining test scores.", '',
          'All RBF runs use the same training split. Comparing coarse and fine HOG within this family tests descriptor resolution, with C tuned separately for each. Comparing a linear model with fine-HOG RBF changes both the features and the decision rule, so that comparison cannot isolate either effect.', '',
          '## Difficult clothing categories', '', '| Category | Recall |', '|---|---:|']
for row in recall.itertuples():
    lines.append(f'| {row.category} | {row.recall:.2%} |')
lines += ['', 'The selected model most often confuses these upper-body categories:', '']
for row in confusions.head(5).itertuples():
    lines.append(f'- {row.true} to {row.predicted}: {row.count} images ({row.rate_within_true_class:.1%} of that true category).')
lines += ['', 'Similar sleeves, shapes and textures at 28x28 resolution may help explain these mistakes. The images alone cannot establish the cause or show that a label is wrong. We use a fixed seed to sample upper-body mistakes for inspection.', '',
          '## Compute and storage', '',
          f"On the shared CPU, the selected model has median single-image latency of {selected.single_ms_median:.3f} ms (p95 {selected.single_ms_p95:.3f} ms); batch throughput is {selected.batch_images_per_second:.1f} images/s. The saved model occupies {selected.model_bytes/1048576:.2f} MiB and stores {int(selected.complexity_count):,} {selected.complexity_kind}.", '',
          'We use two CPU threads and include HOG extraction, scaling and prediction in the timings. File decoding and model loading are excluded. Batch throughput uses 128 images at a time, so its per-image time differs from single-image latency. Support vectors, tree nodes, stored examples and linear weights describe different forms of model capacity.', '',
          '## Comparison with the supplied reference', '',
          f"The reference reported 85.27% for its best classical model on 12,000 training images. Our validation-selected classical model reports {selected.test_accuracy:.2%} on the same public test benchmark after using 48,000 training images. Training size and settings differ, as may the features and classifier family. We therefore cannot attribute the accuracy difference to one change. We keep reference scores separate from our measured results.", '',
          '## Limits and next phase', '',
          'We use one split and seed with a limited search. The public test set also appeared in the reference, so these results describe a familiar benchmark. We have not tested near-duplicates, splits by product identity, new categories, phone photos or score calibration. Phase 2 should reuse the saved split for a small CNN with pooling, dropout and early stopping, then test the effects of architecture and regularization choices separately. Phase 1 contains no CNN training.', '']
(root / 'reports/phase1_findings.md').write_text('\n'.join(lines))

tex = [f"Validation selects {label(frozen['selected'])}. On the official test set, it achieves {selected.test_accuracy*100:.2f}\\% accuracy and macro F1 {selected.test_macro_f1:.4f}. Among the eight core models, validation selects {label(core_best['name'])}, whose test accuracy is {core_best.test_accuracy*100:.2f}\\%. We fixed both choices before examining this experiment's test scores.",
       '\\begin{table}[t]\\centering\\caption{Test results using 48,000 training images. An asterisk marks an additional RBF experiment.}\\label{tab:results}',
       '\\small\\begin{tabular}{lrr}\\toprule Model & Accuracy (\\%) & Macro F1 \\\\ \\midrule']
for row in results.itertuples():
    suffix = '*' if row.family == 'rbf_svm' else ''
    tex.append(f'{label(row.name)}{suffix} & {row.test_accuracy*100:.2f} & {row.test_macro_f1:.4f} \\\\')
tex += ['\\bottomrule\\end{tabular}\\end{table}',
        '\\begin{figure}[t]\\centering\\includegraphics[width=\\columnwidth]{selected_confusion.png}\\caption{Official test confusion matrix for the selected model. Cells show counts, with shading normalized within each true class.}\\end{figure}',
        '\\begin{table}[t]\\centering\\caption{Recall by category for the model selected on validation data.}\\begin{tabular}{lr}\\toprule Category & Recall (\\%) \\\\ \\midrule']
for row in recall.itertuples():
    tex.append(f'{row.category} & {row.recall*100:.1f} \\\\')
tex += ['\\bottomrule\\end{tabular}\\end{table}']
tex.append('The most frequent upper-body mistakes are ' + '; '.join(f'{r.true} to {r.predicted}: {r.count}' for r in confusions.head(3).itertuples()) + '. Similar shapes and textures at this resolution may contribute to these errors, though we have not established the cause. The notebook includes upper-body examples sampled with a fixed seed.')
tex.append(f'The selected model uses {int(selected.complexity_count):,} {selected.complexity_kind} and {selected.model_bytes/1048576:.2f} MiB of serialized storage. Median single-image latency is {selected.single_ms_median:.3f} ms, with p95 {selected.single_ms_p95:.3f} ms. Batch throughput is {selected.batch_images_per_second:.1f} images/s under the common protocol. These timings exclude file decoding and model loading.')
(root / 'reports/phase1_report/results.tex').write_text('\n'.join(tex)+'\n')
for name in ('selected_confusion.png','error_examples.png'):
    shutil.copyfile(root / 'reports/figures' / name, root / 'reports/phase1_report/images' / name)
readme = root / 'README.md'; text = readme.read_text(); start = text.index('<!-- RESULTS_START -->'); end = text.index('<!-- RESULTS_END -->')
text = text[:start] + '<!-- RESULTS_START -->\n' + f"Validation selects {label(frozen['selected'])}. It achieves {selected.test_accuracy:.2%} accuracy and macro F1 {selected.test_macro_f1:.4f} on the official test set after training on all 48,000 images. See the results, errors and timings in [Phase 1 findings](reports/phase1_findings.md).\n" + text[end:]; readme.write_text(text)
payload = {'core':[{**r,'label':label(r['name'])} for r in core.to_dict('records')],
           'selected':selected.to_dict(), 'selectedLabel':label(frozen['selected']),
           'recall':recall.to_dict('records'), 'confusions':confusions.to_dict('records')}
(root / 'reports/.presentation-build').mkdir(parents=True, exist_ok=True)
(root / 'reports/.presentation-build/slide_data.json').write_text(json.dumps(payload,indent=2))
print('Findings, LaTeX result tables, README scores and slide data generated from final evidence.')
