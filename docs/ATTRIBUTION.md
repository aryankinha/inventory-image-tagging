# Reference and data attribution

This project was developed using the user-supplied `Reference_Fashion_Classifier_AML` as a reference. Its training-only loader, saved split, pipeline scaling, validation selection, and frozen-test workflow informed our implementation. Shared facts include the official class mapping, archive checksums, seed 42, and the 48,000/12,000 split. We reorganized and rewrote the code for Phase 1 using established algorithms.

Our experiment trains on all 48,000 images. It separates the four classifier builders from feature extraction and uses validation to test additional RBF SVMs and finer HOG cells. We trained the models and produced the figures for this experiment. The reference's results and team details remain separate from our Phase 1 work, which excludes its CNN and upload service.


Fashion-MNIST: Han Xiao, Kashif Rasul, Roland Vollgraf, Zalando Research (2017). Original repository: https://github.com/zalandoresearch/fashion-mnist . The publisher's MIT notice is retained in `docs/FASHION_MNIST_LICENSE.txt`.

Methodology resources are listed in `literature_review.md` and the report. scikit-learn and scikit-image implement the established algorithms; this project does not implement those solvers from scratch.

The updated reference also informed the Phase 1 additions: explicit alpha-background
handling and blank-photo rejection, a training-only sample gallery, paired
correctness/bootstrap analysis, broader artifact checking and a Mac setup helper.
These ideas were adapted to our Streamlit app and existing model contracts. The
gallery was generated from our own saved training IDs. No reference model weights,
CNN results, Gradio interface or Phase 2 ablation code were imported.
