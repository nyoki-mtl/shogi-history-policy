"""Replay legal USI moves and encode history from the current player's view."""

import collections
import math
from dataclasses import dataclass

import numpy as np
import rsshogi

START_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
HAND_KINDS = ("PAWN", "LANCE", "KNIGHT", "SILVER", "BISHOP", "ROOK", "GOLD")


@dataclass(frozen=True)
class Conditions:
    ratings: tuple[float, float] | None = None
    clocks: tuple[float, float] | None = None

    def __post_init__(self):
        for name in ("ratings", "clocks"):
            pair = getattr(self, name)
            if pair is not None and (len(pair) != 2 or any(not math.isfinite(x) or x < 0 for x in pair)):
                raise ValueError(f"{name} must be a finite nonnegative (self, opponent) pair, or None")


@dataclass(frozen=True)
class Position:
    initial_sfen: str = START_SFEN
    moves: tuple[str, ...] = ()

    @classmethod
    def from_usi(cls, text):
        tokens = text.strip().split()
        if tokens and tokens[0] == "position":
            tokens = tokens[1:]
        if not tokens:
            raise ValueError("empty position")
        if tokens[0] == "startpos":
            initial, tail = START_SFEN, tokens[1:]
        elif tokens[0] == "sfen" and len(tokens) >= 5:
            initial, tail = " ".join(tokens[1:5]), tokens[5:]
        else:
            raise ValueError("expected startpos or sfen followed by four SFEN fields")
        if tail and tail[0] != "moves":
            raise ValueError("expected moves after initial position")
        return cls(initial, tuple(tail[1:]))


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
            move = rsshogi.core.Move.from_usi(position.moves[i])
            if not board.is_legal_move(move):
                raise ValueError(f"illegal move at index {i}: {position.moves[i]}")
            board.apply_move(move)
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
