from pathlib import Path

import numpy as np
import pytest
from rsshogi.record import Record

from shogi_human_policy import Position
from shogi_human_policy.cli import main
from shogi_human_policy.position import START_SFEN, encode

SAMPLE = Path(__file__).parents[1] / "examples/records/opening.kif"
MOVES = ("7g7f", "3c3d", "2g2f", "8c8d")


def test_bare_sfen_and_usi_round_trip():
    variants = [
        START_SFEN + " moves 7g7f 3c3d",
        "sfen " + START_SFEN + " moves 7g7f 3c3d",
        "position startpos moves 7g7f 3c3d",
    ]
    expected = Position.from_sfen(START_SFEN, moves="7g7f 3c3d")
    for text in variants:
        p = Position.from_usi(text)
        assert p == expected
        assert Position.from_usi(p.to_usi()) == p
        assert p.to_sfen().split()[3] == "3"


def test_absolute_sfen_ply_is_not_reset():
    p = Position.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 100", moves="5i6i 5a6a")
    assert p.to_sfen().split()[3] == "102"
    assert encode(p)[0]["game_ply"] == 102
    assert p.at_ply(1).to_sfen().split()[3] == "101"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp932"])
def test_kif_encodings_and_history(tmp_path, encoding):
    path = tmp_path / "棋譜.kifu"
    path.write_bytes(SAMPLE.read_text(encoding="utf-8").encode(encoding))
    actual = Position.from_file(path, ply=3)
    expected = Position(moves=MOVES[:3])
    assert actual == expected
    assert int(actual.to_board().turn) == 1
    af, al = encode(actual)
    ef, el = encode(expected)
    assert al == el
    for name in af:
        np.testing.assert_array_equal(af[name], ef[name])


def test_kif_terminal_and_zero_ply():
    text = SAMPLE.read_text(encoding="utf-8")
    assert Position.from_kif(text).moves == MOVES
    assert Position.from_kif(text, ply=0) == Position()
    assert Position.from_kif(text).at_ply(2) == Position(moves=MOVES[:2])


def test_kif_result_summary_without_numbered_terminal():
    text = SAMPLE.read_text(encoding="utf-8").split("   5 投了")[0] + "まで4手で後手の勝ち\n"
    assert Position.from_kif(text).moves == MOVES


def test_explicit_file_encoding(tmp_path):
    path = tmp_path / "utf16.kif"
    path.write_bytes(SAMPLE.read_text(encoding="utf-8").encode("utf-16"))
    assert Position.from_file(path, encoding="utf-16").moves == MOVES


def test_capture_promotion_drop_and_same_square_notation():
    text = "手合割：平手\n手数----指手---------消費時間--\n1 ７六歩(77)\n2 ３四歩(33)\n3 ２二角成(88)\n4 同　銀(31)\n5 ４五角打\n"
    actual = Position.from_kif(text)
    expected = Position(moves=("7g7f", "3c3d", "8h2b+", "3a2b", "B*4e"))
    assert actual == expected
    left, _ = encode(actual)
    right, _ = encode(expected)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_kif_variation_is_not_appended():
    text = SAMPLE.read_text(encoding="utf-8") + "\n変化：3手\n   3 ２二角成(88)\n   4 同　銀(31)\n"
    assert Position.from_kif(text).moves == MOVES


def test_handicap_and_white_start():
    text = "手合割：飛車落ち\n手数----指手---------消費時間--\n1 ３四歩(33)\n"
    root = Position.from_kif(text, ply=0)
    assert int(root.to_board().turn) == 1
    final = Position.from_kif(text)
    assert final.moves == ("3c3d",)
    assert int(final.to_board().turn) == 0
    assert final.to_sfen().split()[3] == "2"


def test_kif_board_diagram_and_hands():
    sfen = "4k4/9/9/9/9/9/9/9/4K4 b Pp 1"
    record = Record.from_usi_position("sfen " + sfen + " moves P*5e")
    position = Position.from_kif(record.to_kif())
    assert position.initial_sfen == sfen
    assert position.moves == ("P*5e",)


@pytest.mark.parametrize(
    "suffix,method",
    [(".ki2", "to_ki2"), (".csa", "to_csa"), (".sfen", "to_usi_position"), (".usi", "to_usi_position")],
)
def test_other_record_files(tmp_path, suffix, method):
    record = Record.from_kif_str(SAMPLE.read_text(encoding="utf-8"))
    path = tmp_path / ("sample" + suffix)
    path.write_text(getattr(record, method)(), encoding="utf-8")
    assert Position.from_file(path, ply=2) == Position(moves=MOVES[:2])


@pytest.mark.parametrize("ply", [-1, 5, True])
def test_out_of_range_ply(ply):
    with pytest.raises(ValueError, match="ply"):
        Position.from_file(SAMPLE, ply=ply)


def test_invalid_inputs(tmp_path):
    with pytest.raises(ValueError):
        Position.from_kif("garbage")
    with pytest.raises(ValueError, match="could not be read"):
        Position.from_kif("手合割：平手\n手数----指手---------消費時間--\n1 foobar\n")
    with pytest.raises(ValueError):
        Position.from_sfen(START_SFEN.replace("- 1", "- 0"))
    path = tmp_path / "empty.kif"
    path.write_text("")
    with pytest.raises(ValueError):
        Position.from_file(path)
    with pytest.raises(ValueError, match="supported files"):
        Position.from_file(tmp_path / "unknown.txt")
    with pytest.raises(ValueError, match="illegal move"):
        Position.from_usi("startpos moves 7g7e").to_board()
    for suffix in (".ki2", ".csa"):
        junk = tmp_path / ("junk" + suffix)
        junk.write_text("garbage")
        with pytest.raises(ValueError):
            Position.from_file(junk)


def test_cli_file_and_text_output(model_directory, capsys):
    assert (
        main(
            ["predict", "--model", str(model_directory), "--file", str(SAMPLE), "--ply", "1", "--top-k", "2"]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "後手の手番（次は2手目）" in text
    assert " 1." in text and "%" in text
    with pytest.raises(SystemExit):
        main(["predict", "--model", str(model_directory), "--file", str(SAMPLE), "--moves", "7g7f"])
    with pytest.raises(SystemExit):
        main(["predict", "--model", str(model_directory), "--position", ""])


def test_japanese_rows_retain_probabilities(model):
    position = Position(moves=MOVES[:1])
    result = model.predict(position, top_k=3)
    rows = result.as_rows(position)
    assert [row["usi"] for row in rows] == [move.usi for move in result.moves]
    assert [row["probability"] for row in rows] == [move.probability for move in result.moves]
    assert all(row["notation"].startswith("△") for row in rows)
