"""Command-line policy prediction from SFEN or game records."""

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
    predict = sub.add_parser("predict", help="SFENまたは棋譜から、次の着手を予測")
    predict.add_argument(
        "--model", type=Path, required=True, help="manifest.jsonとpolicy.safetensorsのディレクトリ"
    )
    source = predict.add_mutually_exclusive_group()
    source.add_argument("--position", help="SFEN、またはUSI position文字列。省略時は平手初期局面")
    source.add_argument("--file", type=Path, help="KIF/KIFU、KI2、CSA、SFEN/USIファイル")
    predict.add_argument("--moves", nargs="+", help="--positionの後に進めるUSI着手列")
    predict.add_argument(
        "--ply", type=int, help="入力の開始局面から何手進めるか。0なら開始局面、省略時は末尾"
    )
    predict.add_argument("--encoding", help="棋譜ファイルの文字コード。省略時はUTF-8、CP932の順に判定")
    predict.add_argument("--format", choices=("text", "json"), default="text", help="表示形式（既定：text）")
    predict.add_argument("--self-rating", type=float, help="現在の手番側のR")
    predict.add_argument("--opponent-rating", type=float, help="相手のR")
    predict.add_argument("--self-time-seconds", type=float, help="手番側の着手前の利用可能秒数")
    predict.add_argument("--opponent-time-seconds", type=float, help="相手の利用可能秒数")
    predict.add_argument("--history-plies", type=int, default=7, help="使う過去の手数（0〜7、既定7）")
    predict.add_argument("--temperature", type=float, default=1.0, help="確率の温度（正の値、既定1）")
    predict.add_argument("--top-k", type=int, default=10, help="表示候補数（既定10）")
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--threads", type=int, default=4, help="CPUスレッド数（既定4）")
    args = parser.parse_args(argv)

    def pair(a, b, name):
        if (a is None) != (b is None):
            parser.error(f"{name}: supply both self and opponent, or omit both")
        return None if a is None else (a, b)

    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.file and args.moves:
        parser.error("--movesは--positionと組み合わせてください")
    if args.encoding and not args.file:
        parser.error("--encodingは--fileと組み合わせてください")
    torch.set_num_threads(args.threads)
    try:
        conditions = Conditions(
            pair(args.self_rating, args.opponent_rating, "ratings"),
            pair(args.self_time_seconds, args.opponent_time_seconds, "clocks"),
        )
        if args.file:
            position = Position.from_file(args.file, ply=args.ply, encoding=args.encoding)
        else:
            position = Position.from_usi("startpos" if args.position is None else args.position)
            if args.moves:
                position = Position(position.initial_sfen, (*position.moves, *args.moves))
            position = position.at_ply(args.ply)
        board = position.to_board()
        model = PolicyModel.from_directory(args.model, device=args.device)
        result = model.predict(
            position,
            conditions,
            history_plies=args.history_plies,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    except (ValueError, OSError, LookupError, FloatingPointError) as error:
        parser.error(str(error))
    if args.format == "json":
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        side = "先手" if int(board.turn) == 0 else "後手"
        print(f"{side}の手番（次は{board.to_sfen().split()[3]}手目）")
        if not result.moves:
            print("合法手がありません。")
        for rank, row in enumerate(result.as_rows(position), 1):
            print(f"{rank:2}. {row['notation']} ({row['usi']})  {row['probability']:.2%}")
    return 0
