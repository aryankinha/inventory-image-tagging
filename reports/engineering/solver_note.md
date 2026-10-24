# Linear SVM solver adjustment before final selection

On the full 48,000-image training partition, the initial StandardScaler + LinearSVC(C=0.1) fit took 299.5 seconds. Its C=1 fit was still running after many minutes and was stopped before completion. The initial trial and execution log are preserved here; they are not part of the final core results.

For the final core comparison both linear-SVM C values are rerun on fixed normalized pixels/HOG without a dataset-level StandardScaler, with tolerance 1e-3 and max_iter=2,000. This avoids amplifying almost-constant border features and reduces conditioning/solver costs. Logistic Regression retains its training-fitted StandardScaler. Convergence warnings still exclude a candidate. Actual fitted settings are stored in models and recorded below in final trial evidence.

This adjustment used runtime and validation evidence only. The official test set was not used to make the change. The same updated linear-SVM recipe is applied to pixels and coarse HOG. It must not be interpreted as a controlled scaler-only comparison because solver tolerance also changed.
