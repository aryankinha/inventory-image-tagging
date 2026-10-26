# Literature review

We compare fixed image representations and classical classifiers in Phase 1. Phase 2 will test whether a small CNN can learn features that distinguish similar clothing categories more accurately.

| Primary source | Method or contribution | Use here | Limitation when interpreting it |
|---|---|---|---|
| Xiao, Rasul and Vollgraf (2017), [Fashion-MNIST](https://arxiv.org/abs/1708.07747) | A common 28x28, ten-class clothing benchmark | Official 60k/10k split, category mapping, and dataset citation | Low-resolution benchmark results do not establish real catalogue performance |
| Dalal and Triggs (2005), [HOG for human detection](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) | Local gradient orientation histograms with contrast normalization | Fixed shape descriptors and interpretable HOG plots | Original task is human detection; HOG does not learn clothing-specific features |
| Cortes and Vapnik (1995), [Support-vector networks](https://doi.org/10.1007/BF00994018) | Maximum-margin decision boundaries | Linear SVM baseline | Linear boundaries may miss complicated category differences |
| Breiman (2001), [Random Forests](https://doi.org/10.1023/A:1010933404324) | Ensembles of randomized decision trees | A nonlinear pixel/HOG classifier | Image locality is not explicitly modeled; fitted forests may be large |
| Cover and Hart (1967), [Nearest neighbor pattern classification](https://doi.org/10.1109/TIT.1967.1053964) | Classification by nearby examples | Distance-weighted k-NN | Retains the training set and can be expensive at prediction time |
| LeCun et al. (1998), [Gradient-based learning applied to document recognition](https://doi.org/10.1109/5.726791) | Convolution, weight sharing, learned visual representations | Faculty pre-read motivating the later CNN phase | It studies document recognition and predates Fashion-MNIST; no Phase 1 CNN result follows from it |

## Research questions

A pixel vector keeps the brightness at every image position, and reshaping recovers the image exactly. Ordinary vector classifiers have no shared local filters. HOG instead describes local edge directions and suppresses some intensity information. Its usefulness depends on the classifier.

We compare pixel and HOG macro F1 within each classifier family and examine the categories each model confuses. We also measure model storage and prediction time. The eight core combinations use the same images, with settings tuned separately for each combination.

We use scikit-learn and scikit-image. Their documentation explains the implementations and settings:

- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): multiclass linear scores fitted with a regularized log-loss objective.
- [Nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html): distance metrics and voting.
- [Forests](https://scikit-learn.org/stable/modules/ensemble.html#forest): bagging and randomized features.
- [SVMs](https://scikit-learn.org/stable/modules/svm.html): margins, C, kernels and computational limits.
- [HOG](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html): descriptor computation and visualization.
- [Preventing leakage](https://scikit-learn.org/stable/common_pitfalls.html): fit transformations only on training data.

The course statement and assessment instructions were supplied locally as PDFs. Details of the implementation are our choices; the PDFs do not demand a particular minimum accuracy or require using a CNN during Phase 1.
