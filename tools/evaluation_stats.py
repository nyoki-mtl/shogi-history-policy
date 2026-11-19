"""Paired uncertainty for aggregate validation comparisons."""

import numpy as np


def game_bootstrap(values, game_ids, *, reference, seed=20261120, repetitions=2000):
    """Resample entire games; retain position-weighted means and paired differences."""
    _, groups = np.unique(game_ids, return_inverse=True)
    counts = np.bincount(groups)
    sums = np.array([np.bincount(groups, weights=row) for row in values])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), repetitions)
    means = (weights @ sums.T) / (weights @ counts)[:, None]
    return dict(
        mean_ci95=np.quantile(means, [0.025, 0.975], axis=0).T.tolist(),
        delta_ci95=np.quantile(means - means[:, reference, None], [0.025, 0.975], axis=0).T.tolist(),
    )
