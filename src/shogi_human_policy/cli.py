"""Local inference CLI. This is not a USI search engine."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .inference import PolicyModel
from .position import Conditions, Position


def main(argv=None):
    parser = argparse.ArgumentParser(prog="shogi-human-policy")
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict", help="predict legal moves for a USI position")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--position", default="startpos")
    predict.add_argument("--self-rating", type=float)
    predict.add_argument("--opponent-rating", type=float)
    predict.add_argument("--self-time-seconds", type=float)
    predict.add_argument("--opponent-time-seconds", type=float)
    predict.add_argument("--history-plies", type=int, default=7)
    predict.add_argument("--temperature", type=float, default=1.0)
    predict.add_argument("--top-k", type=int, default=10)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    def pair(a, b, name):
        if (a is None) != (b is None):
            parser.error(f"{name}: supply both self and opponent, or omit both")
        return None if a is None else (a, b)

    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        conditions = Conditions(
            pair(args.self_rating, args.opponent_rating, "ratings"),
            pair(args.self_time_seconds, args.opponent_time_seconds, "clocks"),
        )
        model = PolicyModel.from_directory(args.model, device=args.device)
        result = model.predict(
            Position.from_usi(args.position),
            conditions,
            history_plies=args.history_plies,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    except (ValueError, FileNotFoundError, FloatingPointError) as error:
        parser.error(str(error))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0
