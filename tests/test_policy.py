import json

import numpy as np
import pytest

from shogi_human_policy import Conditions, PolicyModel, Position
from shogi_human_policy.cli import main
from shogi_human_policy.position import encode


def test_jax_float32_golden(network, reference):
    batch = {k.removeprefix("input:"): v for k, v in reference.items() if k.startswith("input:")}
    actual = network(batch).numpy()
    np.testing.assert_allclose(actual, reference["expected"], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("mask", [[0, 0], [1, 0], [0, 1]])
def test_unknown_values_are_masked(network, reference, mask):
    a = {k.removeprefix("input:"): v[:4].copy() for k, v in reference.items() if k.startswith("input:")}
    a["condition_known"][:] = mask
    expected = network(a).numpy()
    if mask[0] == 0:
        a["ratings"][:] = 9999
    if mask[1] == 0:
        a["clock_seconds"][:] = 99999
    np.testing.assert_array_equal(network(a).numpy(), expected)


def test_white_orientation_and_history():
    black, legal = encode(Position())
    white, _ = encode(Position.from_usi("position startpos moves 7g7f"))
    assert len(legal) == 30
    expected = black["board_piece_history"][::-1, 0]
    expected = np.where(expected == 0, 0, np.where(expected < 16, expected + 16, expected - 16))
    np.testing.assert_array_equal(white["board_piece_history"][:, 1], expected)
    assert white["game_ply"] == 2
    cut, _ = encode(Position.from_usi("startpos moves 7g7f"), history_plies=0)
    assert not cut["board_piece_history"][:, 1:].any()
    assert (cut["hand_count_history"][:, 1:] == 31).all()


def test_repetition_is_recomputed_after_cut():
    position = Position.from_usi("sfen 4k4/9/9/9/9/9/9/9/4K4 b - 100 moves 5i6i 5a6a 6i5i 6a5a")
    full, _ = encode(position)
    cut, _ = encode(position, history_plies=0)
    assert full["repetition_history"][0] == np.float32(1 / 3)
    assert cut["repetition_history"][0] == 0
    assert cut["game_ply"] == 104


def test_missing_is_distinct_from_known_zero():
    unknown, _ = encode(Position())
    known, _ = encode(Position(), Conditions(ratings=(0, 0), clocks=(0, 0)))
    np.testing.assert_array_equal(unknown["ratings"], known["ratings"])
    np.testing.assert_array_equal(unknown["condition_known"], [0, 0])
    np.testing.assert_array_equal(known["condition_known"], [1, 1])


def test_legal_distribution_and_top_k(model):
    position = Position.from_usi("startpos moves 7g7f 3c3d")
    full = model.predict(position)
    short = model.predict(position, top_k=3)
    assert sum(x.probability for x in full.moves) == pytest.approx(1)
    assert short.moves == full.moves[:3]
    assert sum(x.probability for x in short.moves) < 1
    assert full.legal_count == 39
    assert full.entropy > 0
    for result in model.predict_batch([position, position]):
        assert [x.usi for x in result.moves] == [x.usi for x in full.moves]
        np.testing.assert_allclose(
            [x.probability for x in result.moves], [x.probability for x in full.moves], atol=1e-7
        )


@pytest.mark.parametrize("text", ["", "startpos foo", "startpos moves 7g7e"])
def test_invalid_positions(text):
    with pytest.raises(ValueError):
        encode(Position.from_usi(text))


@pytest.mark.parametrize("kwargs", [{"ratings": (-1, 0)}, {"clocks": (float("nan"), 0)}, {"ratings": (1,)}])
def test_invalid_conditions(kwargs):
    with pytest.raises(ValueError):
        Conditions(**kwargs)


def test_corrupt_model_is_rejected(model_directory):
    with (model_directory / "policy.safetensors").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA256"):
        PolicyModel.from_directory(model_directory)


def test_no_legal_moves(model):
    position = Position(initial_sfen="4k4/4G4/4R4/9/9/9/9/9/4K4 w - 1")
    result = model.predict(position)
    assert result.moves == ()
    assert result.entropy == 0
    assert result.legal_count == 0


@pytest.mark.parametrize(
    "kwargs", [{"temperature": 0}, {"temperature": float("nan")}, {"top_k": 0}, {"history_plies": 8}]
)
def test_invalid_inference_parameters(model, kwargs):
    with pytest.raises(ValueError):
        model.predict(Position(), **kwargs)


def test_cli(model_directory, capsys):
    assert main(["predict", "--model", str(model_directory), "--top-k", "2", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["legal_count"] == 30
    assert len(payload["moves"]) == 2
    with pytest.raises(SystemExit):
        main(["predict", "--model", str(model_directory), "--self-rating", "1500"])
