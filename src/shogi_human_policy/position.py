"""Replay legal USI moves and encode history from the current player's view."""

import collections
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rsshogi

START_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
HAND_KINDS = ("PAWN", "LANCE", "KNIGHT", "SILVER", "BISHOP", "ROOK", "GOLD")
_KIF_HEADER = re.compile(
    r"(?m)^\s*(?:#KIF|手合割[：:]|手数[-ー]|先手[：:]|後手[：:]|上手[：:]|下手[：:]|[先後]手の持駒[：:]|\d+\s+\S)"
)


@dataclass(frozen=True)
class Conditions:
    """Optional (side-to-move, opponent) rating and available-time pairs."""

    ratings: tuple[float, float] | None = None
    clocks: tuple[float, float] | None = None

    def __post_init__(self):
        for name in ("ratings", "clocks"):
            pair = getattr(self, name)
            if pair is not None and (len(pair) != 2 or any(not math.isfinite(x) or x < 0 for x in pair)):
                raise ValueError(f"{name} must be a finite nonnegative (self, opponent) pair, or None")


@dataclass(frozen=True)
class Position:
    """Initial SFEN and a main-line move prefix, retaining history for inference."""

    initial_sfen: str = START_SFEN
    moves: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(
            self, "moves", tuple(self.moves.split() if isinstance(self.moves, str) else self.moves)
        )

    @classmethod
    def from_sfen(cls, sfen: str, moves=()) -> "Position":
        """Create a position from four SFEN fields and optional USI moves."""
        fields = sfen.strip().split()
        if len(fields) != 4 or int(fields[3]) < 1:
            raise ValueError("SFEN requires four fields and a positive move number")
        initial = rsshogi.core.Board(sfen=" ".join(fields)).to_sfen()
        return cls(initial, moves)

    @classmethod
    def from_usi(cls, text: str) -> "Position":
        """Accept a USI position command, bare SFEN, or either with moves."""
        tokens = text.strip().split()
        if tokens and tokens[0] == "position":
            tokens = tokens[1:]
        if not tokens:
            raise ValueError("empty position")
        if "/" in tokens[0]:
            tokens.insert(0, "sfen")
        if tokens[0] == "startpos":
            initial, tail = START_SFEN, tokens[1:]
        elif tokens[0] == "sfen" and len(tokens) >= 5:
            initial, tail = " ".join(tokens[1:5]), tokens[5:]
        else:
            raise ValueError("expected startpos or sfen followed by four SFEN fields")
        if tail and tail[0] != "moves":
            raise ValueError("expected moves after initial position")
        return cls.from_sfen(initial, tail[1:])

    @classmethod
    def from_kif(cls, text: str, *, ply: int | None = None) -> "Position":
        """Read KIF main line; ply is the number of moves applied from its root."""
        from rsshogi.record import Record

        text = text.lstrip("\ufeff")
        # The native reader can silently skip unrecognized lines.
        if not _KIF_HEADER.search(text):
            raise ValueError("KIF headers or numbered moves were not found")
        record = Record.from_kif_str(text)
        main_line = re.split(r"(?m)^\s*変化[：:]", text, maxsplit=1)[0]
        numbered_rows = [
            body
            for number, body in re.findall(r"(?m)^\s*(\d+)[ \t　]+([^\r\n]+)", main_line)
            if int(number) > 0 and re.search(r"[^\d\s]", body)
        ]
        extra_rows = len(numbered_rows) - record.move_count
        if extra_rows not in (0, 1) or (extra_rows == 1 and record.main_terminal is None):
            raise ValueError("KIF contains numbered moves that could not be read")
        return cls._from_record(record, ply)

    @classmethod
    def from_file(
        cls, path: str | Path, *, ply: int | None = None, encoding: str | None = None
    ) -> "Position":
        """Read KIF/KIFU, KI2, CSA, or a text SFEN/USI position file."""
        from rsshogi.record import Record

        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in (".kif", ".kifu", ".ki2", ".csa", ".sfen", ".usi"):
            raise ValueError("supported files: .kif, .kifu, .ki2, .csa, .sfen, .usi")
        if encoding is None:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                text = path.read_text(encoding="cp932")
        else:
            text = path.read_text(encoding=encoding).lstrip("\ufeff")
        if not text.strip():
            raise ValueError("position file is empty")
        if suffix in (".sfen", ".usi"):
            return cls.from_usi(text).at_ply(ply)
        if suffix in (".kif", ".kifu"):
            return cls.from_kif(text, ply=ply)
        if suffix == ".ki2" and not (_KIF_HEADER.search(text) or re.search("[▲△☗☖]", text)):
            raise ValueError("KI2 headers or moves were not found")
        if suffix == ".csa" and not re.search(r"(?m)^\s*(?:V\d|P|[+-]|N[+-])", text):
            raise ValueError("CSA headers or moves were not found")
        reader = Record.from_ki2_str if suffix == ".ki2" else Record.from_csa_str
        return cls._from_record(reader(text), ply)

    @classmethod
    def _from_record(cls, record, ply):
        if record.main_terminal is not None and record.main_terminal.kind == "UNKNOWN":
            raise ValueError("record contains a terminal or move that could not be read")
        return cls.from_sfen(
            record.init_position_sfen, [entry.move.to_usi() for entry in record.moves]
        ).at_ply(ply)

    def at_ply(self, ply: int | None) -> "Position":
        """Select the position after ply moves, with None selecting the end."""
        if ply is None:
            return self
        if type(ply) is not int or not 0 <= ply <= len(self.moves):
            raise ValueError(f"ply must be between 0 and {len(self.moves)}")
        return Position(self.initial_sfen, self.moves[:ply])

    def to_usi(self) -> str:
        initial = "startpos" if self.initial_sfen == START_SFEN else "sfen " + self.initial_sfen
        return "position " + initial + (" moves " + " ".join(self.moves) if self.moves else "")

    def to_board(self):
        """Replay the selected prefix and return an independent native board."""
        board = rsshogi.core.Board(sfen=self.initial_sfen)
        for index, usi in enumerate(self.moves):
            _apply_legal(board, usi, index)
        return board

    def to_sfen(self) -> str:
        return self.to_board().to_sfen()


def _apply_legal(board, usi, index):
    move = rsshogi.core.Move.from_usi(usi)
    if not board.is_legal_move(move):
        raise ValueError(f"illegal move at index {index}: {usi}")
    board.apply_move(move)


def encode(position, conditions=None, *, history_plies=7):
    conditions = conditions if conditions is not None else Conditions()
    if type(history_plies) is not int or not 0 <= history_plies <= 7:
        raise ValueError("history_plies must be between 0 and 7")
    board = rsshogi.core.Board(sfen=position.initial_sfen)
    initial_ply = int(position.initial_sfen.split()[3])
    if initial_ply < 1:
        raise ValueError("SFEN move number must be positive")
    frames = collections.deque(maxlen=8)
    seen = collections.Counter()
    for i in range(len(position.moves) + 1):
        state = board.to_position_state()
        key = board.zobrist_hash()
        hand = np.array(
            [[h.count(getattr(rsshogi.types.PieceType, k)) for k in HAND_KINDS] for h in state.hands],
            np.int32,
        )
        frames.append((np.array(state.board, np.int32), hand, key, min(seen[key], 3) / 3))
        seen[key] += 1
        if i < len(position.moves):
            _apply_legal(board, position.moves[i], i)
    frames = list(frames)[-(history_plies + 1) :]
    if history_plies < 7:
        recent = collections.Counter()
        cut = []
        for pieces, hand, key, _ in frames:
            cut.append((pieces, hand, key, min(recent[key], 3) / 3))
            recent[key] += 1
        frames = cut
    boards, hands, repetitions = [], [], []
    black = int(board.turn) == 0
    for pieces, hand, _, repeat in reversed(frames):
        if not black:
            pieces = pieces[::-1]
            pieces = np.where(pieces == 0, 0, np.where(pieces < 16, pieces + 16, pieces - 16))
            hand = hand[::-1]
        boards.append(pieces)
        hands.append(np.minimum(hand.reshape(-1), 18) + 31)
        repetitions.append(repeat)
    while len(boards) < 8:
        boards.append(np.zeros(81, np.int32))
        hands.append(np.full(14, 31, np.int32))
        repetitions.append(0.0)
    legal = {}
    for move in board.legal_moves():
        label = rsshogi.policy.compact_move_label(move, board.turn)
        if label is None or label in legal:
            raise ValueError("legal policy label collision")
        legal[label] = move.to_usi()
    return dict(
        board_piece_history=np.asarray(boards, np.int64).T,
        hand_count_history=np.asarray(hands, np.int64).T,
        repetition_history=np.asarray(repetitions, np.float32),
        game_ply=np.int64(initial_ply + len(position.moves)),
        draw_limit_max_plies=np.int64(320),
        ratings=np.asarray(conditions.ratings or (0, 0), np.float32),
        clock_seconds=np.asarray(conditions.clocks or (0, 0), np.float32),
        condition_known=np.array([conditions.ratings is not None, conditions.clocks is not None], np.float32),
    ), legal
