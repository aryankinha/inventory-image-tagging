# Phase 2 viva guide

Every member should be able to answer these in plain words. Numbers are in `reports/phase2/` and `reports/phase2_findings.md`; look them up, do not memorise a different set.

## One-minute explanation

Phase 1 turned each picture into a fixed list of numbers (pixels or HOG edges) and handed it to a classical classifier. A CNN learns that list itself. We trained a small CNN on the same 48,000 pictures, chose its settings on the same 12,000 validation pictures, and tested once on the official 10,000. Then we tested a hybrid: the CNN's learned 64 numbers go into a Phase 1 classifier instead of the CNN's last layer.

## Code reading order

1. `src/inventory_tagging_dl/cnn.py`: the model, the augmentation, the training loop with early stopping.
2. `src/inventory_tagging_dl/experiments.py`: ablation, selection, hybrid, final test, figures.
3. `src/inventory_tagging_dl/cli.py`: one command per stage.
4. `tests/test_phase2.py`: what we check automatically.

## Questions to practise

**What does a convolution do?** A small 3x3 grid of weights slides across the picture and measures how strongly each spot matches a pattern (an edge, a corner). The same weights are used everywhere (weight sharing), so the network needs few weights and finds a pattern wherever it appears.

**What does max pooling do?** It keeps the strongest response in each 2x2 patch, halving width and height (28 to 14 to 7). The network becomes less sensitive to a small shift of the pattern, and later layers see a wider part of the picture.

**Why does Phase 1 flattening lose something?** Flattening keeps every pixel but gives the classifier no idea which pixels are neighbours. A convolution uses the neighbourhood directly.

**Why ReLU?** `max(0, x)` is cheap, does not squash large values, and gives gradients that do not vanish for positive inputs.

**What is batch normalisation?** After each convolution it rescales the numbers in the current batch to mean 0 and spread 1, then lets the network learn a scale and shift. It makes training steadier. It acts differently in training (batch statistics) and evaluation (stored averages), which is why we call `model.eval()` before predicting.

**What is dropout?** During training each unit in a layer is switched off with probability 0.4, so the network cannot depend on any single unit. At test time all units are on. It reduces overfitting.

**What is overfitting and how do we see it?** Training loss keeps falling while validation loss rises. Our learning-curve figure shows both.

**What is early stopping?** We watch validation loss and stop after 5 epochs without improvement, then restore the weights of the best epoch. Only validation data is used, never test data.

**What is data augmentation, and why these two?** We randomly mirror and shift training pictures by up to 2 pixels. A mirrored or nudged shirt is still a shirt, so the network sees more variety. We did not rotate or zoom: that is more machinery than the problem needs. Augmentation is applied to training pictures only.

**Why Adam and weight decay?** Adam adapts the step size for each weight and works well without much tuning. Weight decay (0.0001) gently pulls weights toward zero, another way to limit overfitting.

**Why three seeds?** Training starts from random weights, so one run is one draw. Three seeds show how much a result moves by chance. If two settings differ by less than that spread, we do not claim one is better.

**What is an ablation study?** Remove one ingredient at a time from the full model and see what gets worse. We removed batch norm, dropout and augmentation one at a time, tried a third convolution block, ran with all three removed, and added a lighter dropout (0.2) after the first runs showed dropout 0.4 hurt. Be ready to say that last step was decided after looking at validation results, never test results.

**Why is the hybrid "justified"?** The CNN's second-to-last layer holds 64 learned numbers that summarise a picture. A classical classifier on those numbers may draw a better boundary than the CNN's single linear last layer. We test that claim instead of assuming it, with the CNN's own last layer as the comparison and untrained-CNN features as a control.

**Is there any leakage in the hybrid?** The classical classifier is fitted on training-picture features from a CNN that already saw those pictures. Those features may look cleaner than features of new pictures. Validation and test pictures were never used to fit either part.

**How do we know the test comparison is fair?** Same official test pictures for every model, choices frozen in `reports/phase2/frozen.json` before the test, test run once (a marker file blocks a second run), and a paired bootstrap for the difference between two models on the same pictures.

**What does the paired bootstrap tell us?** We resample the 10,000 test pictures many times and recompute the accuracy difference each time. If the 95% interval excludes zero, the difference is unlikely to be just which pictures happened to be in the test set.

**Where does the CNN still fail?** Look at `reports/phase2/test/upper_body_confusions.csv` and `figures/error_examples.png`. Shirt versus T-shirt, pullover and coat is the hard group.

**What would we do next with more time?** Be honest and specific: more seeds, a search over learning rate, testing on real photos. We did not do these.

**Why is the final CNN not the best-scoring version?** The best-scoring version had no dropout, but the project brief asks for a CNN with dropout. We picked the best version that uses it (dropout 0.2). Its score is 0.0018 lower, less than one seed spread. Say this openly.

**Why did dropout 0.4 hurt?** Our reading is that the network overfits little at this size, so strong regularisation removes information it needs. We did not test this further, so say "our reading", not "proved".

## Limits to state yourself before being asked

- Fashion-MNIST is a familiar, small, centred benchmark. Results say little about real store photos.
- Three seeds, one split, a small settings search.
- Training on the Apple GPU is not bit-for-bit repeatable, so a rerun can differ slightly.
- The CNN is not in the live dashboard.
