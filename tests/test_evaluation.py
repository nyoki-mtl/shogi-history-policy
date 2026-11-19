import importlib.util
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
