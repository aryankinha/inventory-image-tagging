"""Four textbook model families; only a small, explicit search is used."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def choices(family):
    if family == 'logistic_regression':
        return [{'C': 0.1}, {'C': 1.0}]
    if family == 'knn':
        return [{'n_neighbors': 3}, {'n_neighbors': 5}]
    if family == 'random_forest':
        return [{'n_estimators': 200, 'max_depth': 25},
                {'n_estimators': 400, 'max_depth': None}]
    if family == 'linear_svm':
        return [{'C': 0.1}, {'C': 1.0}]
    raise ValueError(family)


def build(family, parameters):
    if family == 'logistic_regression':
        estimator = LogisticRegression(max_iter=2000, solver='lbfgs', **parameters)
        return make_pipeline(StandardScaler(), estimator)
    if family == 'linear_svm':
        # Pixels are already in [0,1]; HOG already has block normalization.
        # Avoid amplifying rare, almost-constant border pixels with a feature scaler.
        return LinearSVC(dual=False, tol=1e-3, max_iter=2000, random_state=42, **parameters)
    if family == 'knn':
        return KNeighborsClassifier(weights='distance', metric='euclidean', n_jobs=2, **parameters)
    if family == 'random_forest':
        return RandomForestClassifier(max_features='sqrt', random_state=42, n_jobs=2, **parameters)
    raise ValueError(family)


def complexity(model):
    """Report honest, model-specific capacity measures rather than equivalent counts."""
    estimator = model[-1] if hasattr(model, 'steps') else model
    if hasattr(estimator, 'estimators_'):
        return {'kind': 'tree nodes', 'count': sum(t.tree_.node_count for t in estimator.estimators_),
                'trees': len(estimator.estimators_)}
    if hasattr(estimator, '_fit_X'):
        return {'kind': 'stored training images', 'count': len(estimator._fit_X),
                'stored_feature_bytes': estimator._fit_X.nbytes}
    return {'kind': 'linear coefficients and intercepts',
            'count': estimator.coef_.size + estimator.intercept_.size}
