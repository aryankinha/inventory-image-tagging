# Phase 2 findings

A small CNN beats our best Phase 1 model by about 4 points of test accuracy, and a hybrid (CNN features into a Random Forest) adds about half a point more. Every number below comes from `reports/phase2/`.

## Setup

- Same data as Phase 1: 48,000 training and 12,000 validation pictures from the official training set, then the official 10,000 test pictures.
- Every choice used validation pictures only. The choices were saved in `reports/phase2/frozen.json` before the test, and the test ran once.
- Every CNN setting was trained with three seeds (42, 1, 2). We report the mean and the spread.

## Test results

| Model | Test accuracy | Macro F1 |
|---|---:|---:|
| Phase 1 best: Pixels + Random Forest | 87.61% | 0.8747 |
| CNN (mean of 3 seeds) | 91.98% (±0.24) | 0.9194 (±0.0032) |
| Hybrid: CNN features + Random Forest (mean of 3 seeds) | 92.49% (±0.19) | 0.9246 (±0.0020) |

Paired bootstrap on the same 10,000 pictures (seed 42, 95% interval):

- CNN minus Phase 1 best: +4.10 points, interval [+3.52, +4.67].
- Hybrid minus CNN: +0.60 points, interval [+0.28, +0.94].

Both intervals exclude zero. The second gain is small, and it comes from one seed pair; the three-seed averages are 0.51 points apart.

## Ablation (validation macro F1, mean of 3 seeds)

| Version | Validation macro F1 | Spread (std) |
|---|---:|---:|
| No dropout | 0.9282 | 0.0022 |
| Light dropout 0.2 (**final CNN**) | 0.9264 | 0.0033 |
| Plain (no batch norm, dropout or augmentation) | 0.9258 | 0.0023 |
| No augmentation | 0.9254 | 0.0033 |
| Three convolution blocks | 0.9215 | 0.0049 |
| Full: batch norm + dropout 0.4 + augmentation | 0.9202 | 0.0036 |
| No batch norm | 0.9195 | 0.0018 |

What this says, and does not say:

- All seven versions are within 0.9 points of each other. Most gaps are about the size of the seed-to-seed spread, so we do not rank them finely.
- Dropout 0.4 hurt (full 0.9202 vs no dropout 0.9282). Dropout 0.2 is about the same as none. The network is not overfitting much at this size, so strong regularisation costs accuracy.
- Augmentation and batch norm gave no clear gain here. Removing either did not hurt.
- A third convolution block did not help.
- The brief asks for a CNN with dropout. The final CNN is therefore the best version that uses dropout. The best version overall, no dropout, is 0.0018 higher, less than one seed spread.
- Light dropout was added after the first runs showed dropout 0.4 hurt. It was judged on validation data only.

## Hybrid

On validation data, a classifier fitted on the CNN's 64 learned numbers beats the CNN's own last layer:

| Classifier on the 64 numbers | Validation macro F1 |
|---|---:|
| Random Forest, 400 trees (**selected**) | 0.9324 |
| Logistic Regression or linear SVM | 0.931 to 0.932 |
| CNN's own last layer | 0.9264 |
| Same Random Forest on features from an **untrained** CNN | 0.7944 |

The untrained control is far worse, so the training of the CNN, not just its shape, creates the useful features. The choice of classifier matters little: all three families land within 0.15 points of each other. Even Logistic Regression gains about half a point over the CNN's own last layer, so the gain comes from refitting a classifier on the learned features, not from Random Forest in particular. It is real but small.

## Similar clothing categories

Recall on the test set (CNN and hybrid are 3-seed means):

| Category | Phase 1 best | CNN | Hybrid |
|---|---:|---:|---:|
| T-shirt/top | 85.2% | 88.9% | 88.5% |
| Pullover | 80.2% | 88.7% | 89.6% |
| Coat | 82.2% | 87.0% | 88.2% |
| Shirt | 59.2% | 73.6% | 75.7% |

Common upper-body mistakes (seed 42, counts out of 1,000):

| Actual to predicted | Phase 1 | CNN | Hybrid |
|---|---:|---:|---:|
| Shirt to T-shirt/top | 149 | 128 | 95 |
| Shirt to Pullover | 124 | 80 | 51 |
| Pullover to Coat | 116 | 33 | 37 |
| Coat to Pullover | 91 | 60 | 44 |

The CNN fixes most Pullover/Coat confusion. Shirt is still the hardest category: about one in four shirts is wrong. Pictures at 28×28 give little detail, and the images alone cannot show whether a label is wrong. See `reports/phase2/figures/error_examples.png`.

## Cost

Timed on the same Mac CPU with two threads, from a 28×28 picture to a label. Phase 1 timings were taken in an earlier session on the same machine, so read the comparison as a rough guide.

| Model | Median time for one picture | Pictures per second (batches of 128) | Size |
|---|---:|---:|---|
| Phase 1 Pixels + Random Forest | 24.98 ms | 5,120 | 3.26 million tree nodes, 69.96 MiB |
| CNN | 0.17 ms | 11,314 | 429,450 weights, 1.7 MB |
| Hybrid (CNN + 400-tree Random Forest) | 25.88 ms | 3,441 | 428,800 CNN weights + 1.72 million tree nodes |

## Limits

- Fashion-MNIST is small, centred and familiar. These results say little about real store photos.
- One split, three seeds, a small settings search.
- The CNN training used an Apple GPU, which is not bit-for-bit repeatable. A rerun can differ slightly; the saved predictions and checkpoints are the record.
- In the hybrid, the Random Forest is fitted on training-picture features from a CNN that already saw those pictures. That can make those features look cleaner than features of new pictures. Neither part used validation or test pictures to fit.
- Light dropout was added after seeing results, and the final-model rule (best version with dropout) was set after seeing the ablation. Both used validation data only. Test data was not involved in either decision.
- The CNN is not in the live dashboard.
