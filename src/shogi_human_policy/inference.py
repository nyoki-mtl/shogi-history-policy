"""Verified safetensors loading and legal-move probability distributions."""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files as resources
from pathlib import Path

import numpy as np
import rsshogi
from safetensors.torch import load_file

from .network import ModelSpec, PolicyNetwork
from .position import Conditions, Position, encode


@dataclass(frozen=True)
class MoveProbability:
    usi: str
    probability: float
    label: int


@dataclass(frozen=True)
class Prediction:
    moves: tuple[MoveProbability, ...]
    entropy: float
    legal_count: int

    def as_rows(self, position: Position) -> list[dict[str, str | float]]:
        """Return display rows with USI, Japanese notation, and full-distribution probability."""
        board = position.to_board()
        return [
            dict(
                usi=item.usi,
                notation=rsshogi.core.Move.from_usi(item.usi).to_ki2(board) or item.usi,
                probability=item.probability,
            )
            for item in self.moves
        ]


class PolicyModel:
    """Verified CPU float32 policy inference with optional rating/clock conditions."""

    def __init__(self, network, manifest):
        self.network, self.manifest = network, manifest

    @classmethod
    def from_directory(cls, directory, *, device="cpu"):
        """Read a manifest and its hash-verified policy.safetensors."""
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "shogi-human-policy-v1"
            or manifest.get("policy_labels") != 1496
            or manifest.get("selection") != "ema"
            or manifest.get("history_frames") != 8
            or manifest.get("draw_reference_plies") != 320
        ):
            raise ValueError("unsupported model contract")
        if (
            manifest.get("conditioning")
            != dict(
                rating_order=["self", "opponent"],
                rating_center=1500,
                rating_scale=1500,
                time_encoding="log1p(seconds)/log1p(3600)",
                missing="paired rating and clock masks after normalization",
            )
            or manifest.get("history_padding") != "zero_board_hand31"
        ):
            raise ValueError("model feature encoding differs")
        if set(manifest.get("files", {})) != {"policy.safetensors"}:
            raise ValueError("unsupported model files")
        path = directory / "policy.safetensors"
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != manifest["files"][path.name]:
                raise ValueError("model SHA256 mismatch")
        geometry = resources("shogi_human_policy").joinpath("geometry.npz").read_bytes()
        if manifest.get("geometry_sha256") != hashlib.sha256(geometry).hexdigest():
            raise ValueError("policy label geometry differs")
        network = PolicyNetwork(
            ModelSpec.from_dict(manifest["architecture"]), load_file(str(path)), device=device
        )
        return cls(network, manifest)

    def predict(
        self,
        position: Position,
        conditions: Conditions | None = None,
        *,
        history_plies: int = 7,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Prediction:
        """Predict legal moves; top-k probabilities retain the full legal-move denominator."""
        conditions = conditions if conditions is not None else Conditions()
        return self.predict_batch(
            [position], [conditions], history_plies=history_plies, temperature=temperature, top_k=top_k
        )[0]

    def predict_batch(
        self,
        positions: Sequence[Position],
        conditions: Sequence[Conditions] | None = None,
        *,
        history_plies: int = 7,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> list[Prediction]:
        """Predict a batch in input order, with one condition group per position."""
        if not positions:
            return []
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be positive or None")
        conditions = [Conditions()] * len(positions) if conditions is None else conditions
        if len(conditions) != len(positions):
            raise ValueError("one condition group is required per position")
        samples = [
            encode(p, c, history_plies=history_plies) for p, c in zip(positions, conditions, strict=True)
        ]
        batch = {k: np.stack([s[0][k] for s in samples]) for k in samples[0][0]}
        logits = self.network(batch).cpu().numpy().astype(np.float64)
        if not np.isfinite(logits).all():
            raise FloatingPointError("nonfinite prediction")
        results = []
        for z, (_, legal) in zip(logits, samples, strict=True):
            labels = np.array(sorted(legal), np.int64)
            if not len(labels):
                results.append(Prediction((), 0.0, 0))
                continue
            selected = z[labels] / temperature
            selected -= selected.max()
            probabilities = np.exp(selected)
            probabilities /= probabilities.sum()
            order = np.argsort(-probabilities, kind="stable")
            entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny))))
            moves = tuple(
                MoveProbability(legal[int(labels[i])], float(probabilities[i]), int(labels[i]))
                for i in order[:top_k]
            )
            results.append(Prediction(moves, entropy, len(legal)))
        return results
