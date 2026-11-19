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
    ratings = np.array([100] * 10 + [700] * 10 + [3200, 3300, 3400])
    known = np.ones(len(ratings), bool)
    known[-1] = False
    indices, bands = module.stratified_indices(ratings, known, 3, 77)
    repeated, _ = module.stratified_indices(ratings, known, 3, 77)
    np.testing.assert_array_equal(indices, repeated)
    assert len(indices) == len(set(indices)) == 8
    assert known[indices].all()
    assert [(bands[indices] == b).sum() for b in [0, 1, 6]] == [3, 3, 2]
