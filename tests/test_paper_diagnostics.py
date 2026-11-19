import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

tools = Path(__file__).parents[1] / "tools"
spec = importlib.util.spec_from_file_location("paper_diagnostics", tools / "evaluate_paper_diagnostics.py")
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(tools))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_rank_shape_can_match_while_actual_moves_are_disjoint():
    p = np.array([[0.7, 0.2, 0.1, 0, 0, 0]])
    q = np.array([[0, 0, 0, 0.7, 0.2, 0.1]])
    m = module.pair_metrics(p, q)
    assert m["rank_jsd"][0] == pytest.approx(0)
    assert m["full_jsd"][0] == pytest.approx(1)
    assert m["tv"][0] == pytest.approx(1)
    assert m["changed"][0] == 1
    assert module.jsd_bits(p, p)[0] == pytest.approx(0)


def test_correctness_pattern_uses_low_mid_high_bit_order():
    predictions = [np.eye(3)[indices] for indices in ([0, 1, 2], [0, 0, 1], [0, 2, 1])]
    patterns = module.correctness_pattern(predictions, np.array([0, 1, 0]))
    np.testing.assert_array_equal(patterns, [7, 4, 0])
    # 000 does not imply the three predicted moves are equal.
    assert len({p[2].argmax() for p in predictions}) == 2


def test_calibration_exact_bins_and_weighting():
    result = module.calibration(np.array([0.1, 0.15, 1.0]), np.array([0, 1, 1]))
    assert [entry["n"] for entry in result["bins"]] == [2, 1]
    assert result["bins"][0]["lower"] == 0.1
    assert result["bins"][-1]["upper"] == 1.0
    assert result["ece"] == pytest.approx(0.25)
