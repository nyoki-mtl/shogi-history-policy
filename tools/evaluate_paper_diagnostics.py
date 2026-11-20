"""Paper-aligned correctness patterns, rank/action JSD, phase response and calibration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from evaluate import FEATURES, metrics, probabilities
from evaluation_stats import game_bootstrap

from shogi_human_policy import PolicyModel

RATINGS = (900, 1600, 2100)
CONTEXTS = ("full", "opponent_fixed", "current_only")
PHASES = ("0〜19手", "20〜39手", "40〜79手", "80手以降")
ACTUAL_CONTEXTS = ("observed", "observed_opponent_fixed", "observed_current_only")


def jsd_bits(p, q):
    midpoint = (p + q) / 2
    tiny = np.finfo(float).tiny
    return np.maximum(
        0,
        (
            np.sum(p * np.log2(np.maximum(p, tiny) / np.maximum(midpoint, tiny)), axis=1)
            + np.sum(q * np.log2(np.maximum(q, tiny) / np.maximum(midpoint, tiny)), axis=1)
        )
        / 2,
    )


def pair_metrics(p, q):
    first = np.argsort(-p, axis=1, kind="stable")[:, :5]
    second = np.argsort(-q, axis=1, kind="stable")[:, :5]
    a = np.take_along_axis(p, first, axis=1)
    b = np.take_along_axis(q, second, axis=1)
    return dict(
        rank_jsd=jsd_bits(a / a.sum(1, keepdims=True), b / b.sum(1, keepdims=True)),
        full_jsd=jsd_bits(p, q),
        tv=np.abs(p - q).sum(1) / 2,
        changed=(p.argmax(1) != q.argmax(1)).astype(float),
    )


def correctness_pattern(predictions, labels):
    correct = np.stack([p.argmax(1) == labels for p in predictions])
    return correct.T @ np.array([4, 2, 1])


def calibration(confidence, correct):
    bins = np.minimum((confidence * 10).astype(int), 9)
    entries = []
    for b in range(10):
        mask = bins == b
        if mask.any():
            entries.append(
                dict(
                    lower=b / 10,
                    upper=(b + 1) / 10,
                    n=int(mask.sum()),
                    confidence=float(confidence[mask].mean()),
                    accuracy=float(correct[mask].mean()),
                )
            )
    ece = sum(e["n"] * abs(e["confidence"] - e["accuracy"]) for e in entries) / len(confidence)
    return dict(ece=ece, bins=entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Private per-position metric cache")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(0.15)
    model = PolicyModel.from_directory(args.model, device=args.device)
    with np.load(args.dataset, allow_pickle=False) as a:
        data = {k: a[k] for k in (*FEATURES, "legal_policy_mask", "labels", "game_id", "local_ply")}
    with args.dataset.open("rb") as stream:
        identity = hashlib.file_digest(stream, "sha256").hexdigest()
    n = len(data["labels"])
    phase = np.digitize(data["local_ply"], [20, 40, 80])
    band = (data["ratings"][:, 0] // 200).astype(int)
    if not (
        ((band >= 0) & (band < 15) & (data["condition_known"][:, 0] > 0)).all()
        and data["legal_policy_mask"][np.arange(n), data["labels"]].all()
    ):
        raise ValueError("Invalid matched validation rows")
    precision = "float32 " + args.device.upper() + "; TF32 disabled"
    metadata = dict(
        dataset_sha256=identity,
        weights_sha256=model.manifest["files"]["policy.safetensors"],
        precision=precision,
    )
    values, cursor = {}, 0
    if args.cache.exists():
        with np.load(args.cache, allow_pickle=False) as a:
            if any(str(a[k]) != v for k, v in metadata.items()):
                raise ValueError("Private metric cache differs")
            cursor = int(a["cursor"])
            values = {k: a[k] for k in a.files if k not in {*metadata, "cursor"}}

    def store(key, v, start, end):
        if key not in values:
            values[key] = np.zeros(n, dtype=np.asarray(v).dtype)
        values[key][start:end] = v

    def checkpoint(end):
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, cursor=end, **metadata, **values)

    for start in range(cursor, n, args.batch_size):
        end = min(start + args.batch_size, n)
        source = {k: data[k][start:end] for k in FEATURES}
        legal, labels = data["legal_policy_mask"][start:end], data["labels"][start:end]
        full_predictions = None
        for context in (*CONTEXTS, "observed"):
            predictions = []
            for rating in (None,) if context == "observed" else RATINGS:
                batch = {k: v.copy() for k, v in source.items()}
                if rating is not None:
                    batch["ratings"][:, 0] = rating
                if context == "opponent_fixed":
                    batch["ratings"][:, 1] = 1600
                if context == "current_only":
                    batch["board_piece_history"][:, :, 1:] = 0
                    batch["hand_count_history"][:, :, 1:] = 31
                    batch["repetition_history"][:] = 0
                logits = model.network(batch).cpu().numpy()
                if not np.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite logits")
                p = probabilities(logits, legal)
                predictions.append(p)
                m = metrics(p, labels)
                m["confidence"] = p.max(1)
                m["top5"] = (
                    (np.argsort(-p, axis=1, kind="stable")[:, :5] == labels[:, None]).any(1).astype(float)
                )
                for key, value in m.items():
                    store(f"{context}.{rating}.{key}", value, start, end)
            if context == "observed":
                continue
            store(f"{context}.pattern", correctness_pattern(predictions, labels), start, end)
            for i, j in ((0, 1), (1, 2), (0, 2)):
                for key, value in pair_metrics(predictions[i], predictions[j]).items():
                    store(f"{context}.{RATINGS[i]}_{RATINGS[j]}.{key}", value, start, end)
            if context == "full":
                full_predictions = predictions
            else:
                for rating, p, q in zip(RATINGS, predictions, full_predictions, strict=True):
                    store(f"{context}.{rating}.tv_vs_full", np.abs(p - q).sum(1) / 2, start, end)
                    store(
                        f"{context}.{rating}.changed_vs_full",
                        (p.argmax(1) != q.argmax(1)).astype(float),
                        start,
                        end,
                    )
        if end == n or (end // 2048 != start // 2048):
            checkpoint(end)
            print(f"evaluated {end}/{n}", flush=True)

    # Actual-self controls avoid interpreting a forced R1600 versus observed opponent gap.
    if "observed_current_only.None.nll" not in values or cursor < n:
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            for context in ACTUAL_CONTEXTS[1:]:
                batch = {k: data[k][start:end].copy() for k in FEATURES}
                if context == "observed_opponent_fixed":
                    batch["ratings"][:, 1] = 1600
                else:
                    batch["board_piece_history"][:, :, 1:] = 0
                    batch["hand_count_history"][:, :, 1:] = 31
                    batch["repetition_history"][:] = 0
                logits = model.network(batch).cpu().numpy()
                if not np.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite logits")
                p = probabilities(logits, data["legal_policy_mask"][start:end])
                labels = data["labels"][start:end]
                m = metrics(p, labels)
                m["confidence"] = p.max(1)
                m["top5"] = (
                    (np.argsort(-p, axis=1, kind="stable")[:, :5] == labels[:, None]).any(1).astype(float)
                )
                for key, value in m.items():
                    store(f"{context}.None.{key}", value, start, end)
        checkpoint(n)
        print("actual-self-rating controls complete", flush=True)

    def summary(context, mask):
        if not mask.any():
            return None
        result = dict(n=int(mask.sum()), cases={})
        for rating in (None,) if context in ACTUAL_CONTEXTS else RATINGS:
            prefix = f"{context}.{rating}."
            case = {
                key.removeprefix(prefix): float(value[mask].mean())
                for key, value in values.items()
                if key.startswith(prefix)
            }
            case["ece"] = calibration(values[prefix + "confidence"][mask], values[prefix + "top1"][mask])[
                "ece"
            ]
            result["cases"][str(rating)] = case
        if context in ACTUAL_CONTEXTS:
            prefix = context + ".None."
            result["calibration"] = calibration(
                values[prefix + "confidence"][mask], values[prefix + "top1"][mask]
            )
            return result
        patterns = values[f"{context}.pattern"][mask]
        result["patterns"] = (np.bincount(patterns, minlength=8) / len(patterns)).tolist()
        result["partial_correct"] = float(((patterns != 0) & (patterns != 7)).mean())
        same = (values[f"{context}.900_1600.changed"][mask] == 0) & (
            values[f"{context}.1600_2100.changed"][mask] == 0
        )
        result["all_prediction_same"] = float(same.mean())
        result["same_prediction_jsd"] = {
            key: float(values[f"{context}.900_2100.{key}"][mask][same].mean()) if same.any() else None
            for key in ("rank_jsd", "full_jsd")
        }
        result["pairs"] = {}
        for pair in ("900_1600", "1600_2100", "900_2100"):
            prefix = f"{context}.{pair}."
            result["pairs"][pair] = {
                key.removeprefix(prefix): float(value[mask].mean())
                for key, value in values.items()
                if key.startswith(prefix)
            }
        rank = values[f"{context}.900_2100.rank_jsd"][mask]
        result["rank_jsd_thresholds"] = {str(t): float((rank >= t).mean()) for t in (0.01, 0.02, 0.05)}
        result["small_rank_jsd"] = {
            str(t): dict(
                n=int((rank <= t).sum()),
                all_correct=float((patterns[rank <= t] == 7).mean()) if (rank <= t).any() else None,
                first_move_changed=float(values[f"{context}.900_2100.changed"][mask][rank <= t].mean())
                if (rank <= t).any()
                else None,
            )
            for t in (0.001, 0.01)
        }
        return result

    report = dict(
        schema="shogi-human-policy-paper-diagnostics-v1",
        model_version=model.manifest.get("model_version") or model.manifest.get("status", "unknown"),
        **metadata,
        split="validation",
        n=n,
        ratings=list(RATINGS),
        phases=list(PHASES),
        normalization="all legal moves, temperature=1",
        jsd_log_base=2,
        selection="same 30000 matched rating/ply validation rows; 2000 per rating band",
        environment=dict(torch=torch.__version__, numpy=np.__version__),
        contexts={},
    )
    for context in (*CONTEXTS, *ACTUAL_CONTEXTS):
        report["contexts"][context] = dict(
            overall=summary(context, np.ones(n, bool)),
            phases=[summary(context, phase == p) for p in range(4)],
            bands=[summary(context, band == b) for b in range(15)],
            band_phases=[[summary(context, (band == b) & (phase == p)) for p in range(4)] for b in range(15)],
        )
    differences = np.stack(
        [
            values["current_only.1600.nll"] - values["full.1600.nll"],
            values["opponent_fixed.1600.nll"] - values["full.1600.nll"],
            values["observed_opponent_fixed.None.nll"] - values["observed.None.nll"],
            values["observed_current_only.None.nll"] - values["observed.None.nll"],
            np.zeros(n),
        ]
    )
    report["paired_nll_differences"] = dict(
        names=[
            "current_only_self1600",
            "opponent_fixed_self1600",
            "opponent_fixed_actual_self",
            "current_only_actual_self",
            "reference",
        ],
        means=differences.mean(1).tolist(),
        ci95=game_bootstrap(differences, data["game_id"], reference=4)["mean_ci95"],
        unit="game",
        repetitions=2000,
        seed=20261120,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report["paired_nll_differences"]), flush=True)


if __name__ == "__main__":
    main()
