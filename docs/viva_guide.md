# Explain Phase 1 in simple language

## A one-minute explanation

Our model receives a small grayscale clothing image and predicts one of ten categories. We first use its 784 pixel values, then try HOG features that describe edge directions. We compare four classifiers on both representations using the same 48,000 training images. Another 12,000 images help us choose settings before we evaluate the official 10,000-image test set. We also test RBF SVMs and finer HOG cells. Validation selects fine HOG with RBF SVM, which achieves 90.87% test accuracy and macro F1 0.9084. In Phase 2, we will compare it with a small CNN that learns its own features.

## Code reading order

1. `data.py`: read verified files and split image IDs.
2. `features.py`: divide pixels by 255; flatten or calculate HOG.
3. `models.py`: create four classical models and the optional RBF SVM.
4. `train.py`: try two settings; save the best validation model per combination.
5. `eda.py`: summarize training images and plot the data.
6. `evaluate.py`: freeze model choices, calculate final metrics and timing.
7. `cli.py`: select a stage without editing Python source.

## Questions every member should answer

**What are X and y?** X is an image representation. y is one integer category from 0 to 9. This is single-label multiclass classification.

**Why divide by 255?** The original values are uint8, 0-255. Division puts brightness on 0-1 and avoids inconsistent scale. There are no learned statistics in this operation.

**Does flattening destroy the image?** No. Reshaping 784 values back to 28x28 recovers it exactly. A vector model has no built-in local filters or weight sharing.

**How does HOG work?** HOG measures horizontal and vertical changes in brightness to estimate edge strength and direction. It groups directions into histograms in small cells, normalizes nearby cells in blocks and joins the values into a vector. Seven-pixel cells give 324 features; four-pixel cells give 1,296.

**Why use StandardScaler for Logistic Regression?** It makes feature scales comparable for regularized optimization. The scaler is learned only from the training split and saved inside the model pipeline. HOG block normalization and this dataset-level scaler are different operations.

**How does Logistic Regression classify ten classes?** It computes one linear score per class and fits those scores to labels by minimizing log loss with regularization. This is classification despite the name.

**How does k-NN work?** Find the k closest training vectors and vote using inverse-distance weights. It does almost no fitting. Instead, it stores examples and compares each new image with them. Larger k smooths the vote but does not always improve accuracy.

**How does Random Forest work?** Each tree uses a bootstrap sample of the training data and random feature subsets. The forest combines the trees' predictions. It can fit nonlinear rules without needing a kernel. Increasing tree count can stabilize predictions but costs memory and prediction time.

**What does SVM learn?** Boundaries with a large margin between classes. C balances margin regularization against training errors. The linear model uses one-versus-rest multiclass classification. RBF SVC uses pairwise multiclass classifiers and a nonlinear similarity exp(-gamma ||x-z||^2). Changing from linear to RBF SVM changes the decision rule. If we also change HOG cell size, we cannot attribute the difference to cell size alone.

**What is gamma?** It controls how far the influence of one support vector extends. `scale` is a documented rule computed using the training feature variance. We limit the search to two C values because these fits are expensive.

**Why three data partitions?** Training learns weights, validation chooses settings, and test reports the result after choices are frozen. Choosing the best test score and then calling it the selected model would leak test information.

**What is accuracy?** Correct predictions divided by all images. Random guessing with equally likely classes has expected accuracy 10%.

**What is recall?** Of the images truly belonging to one class, how many did we recover? For shirts, it is correctly predicted shirts divided by all true shirts.

**What is macro F1?** Calculate F1 = 2TP/(2TP+FP+FN) separately for each class and average the ten results. Every class gets equal weight, even if errors differ greatly between categories.

**How do you interpret a confusion matrix?** Rows show true classes and columns show predictions. The diagonal contains correct predictions. A Shirt predicted as Coat and a Coat predicted as Shirt are different mistakes.

**What is a parameter count?** We can count linear weights and intercepts. A forest stores nodes, k-NN stores examples, and kernel SVM stores support vectors/coefficients. These capacity measures are not equivalent. Saved file size offers a common storage comparison, though the file format and compression affect it.

**How is latency measured?** On the same CPU, using warmed models and repeated predictions. We include pixel/HOG extraction and pipeline scaling. We report single-image latency separately from 128-image batch throughput. File loading/decoding is excluded.

**Can a phone photo be classified reliably?** That has not been measured. The prediction command accepts grayscale 28x28 inputs like those in the benchmark. Real photos need separate data and validation.

**Did we invent these algorithms?** No. We use scikit-learn and scikit-image to compare established algorithms under a shared evaluation procedure. The supplied reference helped shape the workflow, and we acknowledge it.

**How did we improve the reference?** We increased the training set from 12,000 to 48,000 images and used validation data to tune a small set of settings. We also tried RBF SVMs and finer HOG cells. Since several factors changed, we cannot attribute the accuracy difference to one of them alone.

**What remains for Phase 2?** We will train a small CNN with two or three blocks, pooling, dropout and early stopping on the same split. We will test architecture and regularization choices separately, then compare its errors with the classical models. Phase 1 includes no CNN result.

## Equations to understand

- Logistic scores: z_c = w_c^T x + b_c; softmax(z)_c = exp(z_c)/sum_j exp(z_j).
- Euclidean distance: d(x,z) = sqrt(sum_j (x_j-z_j)^2).
- RBF similarity: K(x,z) = exp(-gamma * ||x-z||^2).
- Pixel feature count: 28 * 28 = 784.
- HOG dimension: blocks_y * blocks_x * cells_per_block_y * cells_per_block_x * bins.
- Macro F1: (1/10) * sum_c F1_c.

## Training objectives in words

Logistic Regression minimizes the negative log probability assigned to the true classes, plus a penalty on large weights. SVM minimizes a weight penalty plus squared hinge penalties for points inside the margin or on the wrong side. C controls the tradeoff; increasing it is not automatically better. A forest chooses splits using Gini impurity, 1 minus the sum of squared class proportions, then combines randomized trees. k-NN has no comparable learned-weight optimization: its choice of distance and neighbors defines the rule. RBF gamma=scale is calculated from training feature variance, never test variance.
