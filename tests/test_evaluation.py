import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def test_legal_normalization_and_nll():
    path = Path(__file__).parents[1] / "tools/evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logits = np.array([[1000, 1000, 9000], [0, np.log(3), 99]])
    p = module.probabilities(logits, np.array([[True, True, False]] * 2))
    np.testing.assert_allclose(p, [[0.5, 0.5, 0], [0.25, 0.75, 0]])
    result = module.summarize(module.metrics(p, np.array([0, 1])))
    assert result["nll"] == pytest.approx((-np.log(0.5) - np.log(0.75)) / 2)
    assert result["top1"] == 1


def test_rating_matrix_stratification():
    tools = Path(__file__).parents[1] / "tools"
    spec = importlib.util.spec_from_file_location("rating_matrix", tools / "evaluate_rating_matrix.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(tools))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    ratings = np.array([100] * 10 + [700] * 10 + [2800, 2900, 2999, 3000, 3200, -1])
    known = np.ones(len(ratings), bool)
    known[22] = False
    indices, bands = module.stratified_indices(ratings, known, 3, 77)
    repeated, _ = module.stratified_indices(ratings, known, 3, 77)
    np.testing.assert_array_equal(indices, repeated)
    assert len(indices) == len(set(indices)) == 8
    assert known[indices].all()
    assert [(bands[indices] == b).sum() for b in [0, 3, 14]] == [3, 3, 2]
    assert ((ratings[indices] >= 0) & (ratings[indices] < 3000)).all()


def test_rating_groups_half_open_boundaries():
    path = Path(__file__).parents[1] / "tools/evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ratings = np.array([-1, 0, 199, 200, 399, 2800, 2999, 3000, 3200])
    values = {"nll": np.arange(len(ratings), dtype=float), "top1": np.ones(len(ratings))}
    groups = module.rating_groups(ratings, np.ones(len(ratings), bool), values)
    assert [(g["lower"], g["upper"], g["n"]) for g in groups] == [(0, 200, 2), (200, 400, 2), (2800, 3000, 2)]


def test_game_bootstrap_keeps_paired_differences_and_position_weights():
    tools = Path(__file__).parents[1] / "tools"
    spec = importlib.util.spec_from_file_location("high_rating", tools / "evaluation_stats.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(tools))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    # One two-position game and one single-position game; a fixed +2 intervention.
    values = np.array([[0, 0, 3], [2, 2, 5]], float)
    result = module.game_bootstrap(values, [10, 10, 20], reference=0, repetitions=1000)
    np.testing.assert_allclose(result["delta_ci95"], [[0, 0], [2, 2]])
    np.testing.assert_allclose(result["mean_ci95"], [[0, 3], [2, 5]])
    assert result == module.game_bootstrap(values, [10, 10, 20], reference=0, repetitions=1000)
