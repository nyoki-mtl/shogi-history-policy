"""Run after placing a model directory under models/."""

import argparse

from shogi_human_policy import Conditions, PolicyModel, Position

parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()
model = PolicyModel.from_directory(args.model)
position = Position.from_usi("startpos moves 7g7f 3c3d")
result = model.predict(position, Conditions(ratings=(1500, 1800), clocks=(90, 60)), top_k=5)
for move in result.moves:
    print(f"{move.usi}: {move.probability:.3%}")
