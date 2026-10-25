"""Descriptive comparisons of fixed models from their saved predictions only."""
import numpy as np


def paired_outcomes(truth, first, second, repeats=2000, seed=42):
    """Paired image bootstrap; no fitting, selection, or new test-set evaluation.

    The interval describes image-sampling uncertainty for these two fixed models.
    It does not measure uncertainty from training data, settings, or random seeds.
    """
    truth, first, second = (np.asarray(values) for values in (truth, first, second))
    if truth.ndim != 1 or not len(truth) or truth.shape != first.shape or truth.shape != second.shape:
        raise ValueError('The saved labels must describe the same nonempty set of images.')
    if any(not np.isin(values, range(10)).all() for values in (truth, first, second)):
        raise ValueError('Saved labels must use the ten clothing categories.')
    if not isinstance(repeats, int) or repeats < 100:
        raise ValueError('Use at least 100 resamples.')
    a, b = first == truth, second == truth
    differences = a.astype(np.int8) - b.astype(np.int8)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(differences, size=len(truth), replace=True).mean() for _ in range(repeats)])
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        'images': len(truth), 'both_right': int((a & b).sum()),
        'first_only_right': int((a & ~b).sum()), 'second_only_right': int((~a & b).sum()),
        'both_wrong': int((~a & ~b).sum()), 'accuracy_difference': float(differences.mean()),
        'interval_95': [float(low), float(high)], 'resamples': repeats, 'seed': seed,
        'scope': 'Descriptive analysis of saved predictions for fixed models; image sampling only, not training-seed variability. Not used for model selection.',
    }
