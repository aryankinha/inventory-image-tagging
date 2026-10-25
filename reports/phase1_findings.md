# Phase 1 findings

We selected Fine HOG + RBF SVM using validation macro F1 before evaluating the official test set. Its test accuracy is 90.87%, with macro F1 0.9084.

Every model trains on the same 48,000 images. We choose settings using 12,000 validation images, then evaluate on the official 10,000 test images. We test two settings for each of eight core combinations and three additional RBF combinations.

## Test results

| Model | Test accuracy | Macro F1 |
|---|---:|---:|
| Fine HOG + RBF SVM | 90.87% | 0.9084 |
| Pixels + RBF SVM | 89.73% | 0.8971 |
| Pixels + RF | 87.61% | 0.8747 |
| HOG + RBF SVM | 87.70% | 0.8769 |
| Pixels + k-NN | 85.09% | 0.8515 |
| Pixels + LR | 83.96% | 0.8390 |
| Pixels + linear SVM | 83.97% | 0.8373 |
| HOG + LR | 84.55% | 0.8450 |
| HOG + linear SVM | 84.01% | 0.8385 |
| HOG + RF | 84.59% | 0.8455 |
| HOG + k-NN | 83.81% | 0.8382 |

## What HOG changes in the core experiment

- logistic regression: HOG minus pixels in macro F1 = +0.0060.
- knn: HOG minus pixels in macro F1 = -0.0132.
- random forest: HOG minus pixels in macro F1 = -0.0292.
- linear svm: HOG minus pixels in macro F1 = +0.0012.

Each pair uses the same images and classifier family. We tune the settings separately for pixels and HOG, so the difference includes that tuning. HOG does not help every classifier.

## Core models and RBF experiments

Validation selects Pixels + RF among the eight core models. Its test accuracy is 87.61%. Across all experiments, validation selects Fine HOG + RBF SVM, with test accuracy 90.87%. We fixed both choices before examining test scores.

All RBF runs use the same training split. Comparing coarse and fine HOG within this family tests descriptor resolution, with C tuned separately for each. Comparing a linear model with fine-HOG RBF changes both the features and the decision rule, so that comparison cannot isolate either effect.

## Difficult clothing categories

| Category | Recall |
|---|---:|
| T-shirt/top | 85.90% |
| Trouser | 97.80% |
| Pullover | 86.20% |
| Dress | 91.90% |
| Coat | 87.10% |
| Sandal | 96.70% |
| Shirt | 71.40% |
| Sneaker | 96.90% |
| Bag | 98.60% |
| Ankle boot | 96.20% |

The selected model most often confuses these upper-body categories:

- Shirt to T-shirt/top: 116 images (11.6% of that true category).
- T-shirt/top to Shirt: 102 images (10.2% of that true category).
- Shirt to Coat: 66 images (6.6% of that true category).
- Shirt to Pullover: 64 images (6.4% of that true category).
- Pullover to Coat: 58 images (5.8% of that true category).

Similar sleeves, shapes and textures at 28x28 resolution may help explain these mistakes. The images alone cannot establish the cause or show that a label is wrong. We use a fixed seed to sample upper-body mistakes for inspection.

## Compute and storage

On the shared CPU, the selected model has median single-image latency of 10.169 ms (p95 15.732 ms); batch throughput is 59.5 images/s. The saved model occupies 52.74 MiB and stores 17,835 support vectors.

We use two CPU threads and include HOG extraction, scaling and prediction in the timings. File decoding and model loading are excluded. Batch throughput uses 128 images at a time, so its per-image time differs from single-image latency. Support vectors, tree nodes, stored examples and linear weights describe different forms of model capacity.

## Comparison with the supplied reference

The reference reported 85.27% for its best classical model on 12,000 training images. Our validation-selected classical model reports 90.87% on the same public test benchmark after using 48,000 training images. Training size and settings differ, as may the features and classifier family. We therefore cannot attribute the accuracy difference to one change. We keep reference scores separate from our measured results.

## Limits and next phase

We use one split and seed with a limited search. The public test set also appeared in the reference, so these results describe a familiar benchmark. We have not tested near-duplicates, splits by product identity, new categories, phone photos or score calibration. Phase 2 should reuse the saved split for a small CNN with pooling, dropout and early stopping, then test the effects of architecture and regularization choices separately. Phase 1 contains no CNN training.
