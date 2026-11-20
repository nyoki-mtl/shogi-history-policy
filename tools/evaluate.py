"""Aggregate a private preencoded validation sample; never export individual rows."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from shogi_human_policy import PolicyModel

FEATURES = (
    "board_piece_history",
    "hand_count_history",
    "repetition_history",
    "game_ply",
    "draw_limit_max_plies",
    "ratings",
    "clock_seconds",
    "condition_known",
)


def probabilities(logits, legal):
    z = np.where(legal, logits.astype(np.float64), -np.inf)
    z -= np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def metrics(p, labels):
    return {
        "nll": -np.log(np.maximum(p[np.arange(len(p)), labels], np.finfo(float).tiny)),
        "top1": (p.argmax(1) == labels).astype(float),
        "entropy": -(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum(1),
    }


def summarize(m, mask=None):
    if mask is None:
        mask = np.ones(len(m["nll"]), bool)
    return {"n": int(mask.sum()), **{k: float(v[mask].mean()) for k, v in m.items()}}


def rating_groups(ratings, known, values):
    return [
        dict(
            lower=lower,
            upper=lower + 200,
            **summarize(values, known & (ratings >= lower) & (ratings < lower + 200)),
        )
        for lower in range(0, 3000, 200)
        if (known & (ratings >= lower) & (ratings < lower + 200)).any()
    ]


def predict(model, data, indices, mode, batch_size):
    result = []
    for start in range(0, len(indices), batch_size):
        index = indices[start : start + batch_size]
        batch = {k: data[k][index].copy() for k in FEATURES}
        if mode in ("no_rating", "no_clock", "neither"):
            if mode != "no_clock":
                batch["condition_known"][:, 0] = 0
            if mode != "no_rating":
                batch["condition_known"][:, 1] = 0
        if mode == "current_only":
            batch["board_piece_history"][:, :, 1:] = 0
            batch["hand_count_history"][:, :, 1:] = 31
            batch["repetition_history"][:] = 0
        if mode.startswith("self_") or mode.startswith("opponent_"):
            side, value = mode.split("_")
            batch["ratings"][:, int(side == "opponent")] = int(value)
        logits = model.network(batch).cpu().numpy()
        if not np.isfinite(logits).all():
            raise FloatingPointError("nonfinite logits")
        result.append(probabilities(logits, data["legal_policy_mask"][index]))
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--baseline-cache", type=Path, help="Private per-position baseline metrics")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(0.15)
    model = PolicyModel.from_directory(args.model, device=args.device)
    with np.load(args.dataset, allow_pickle=False) as archive:
        data = {k: archive[k] for k in (*FEATURES, "legal_policy_mask", "labels")}
    n = len(data["labels"])
    if not data["legal_policy_mask"][np.arange(n), data["labels"]].all():
        raise ValueError("teacher includes illegal moves")
    with args.dataset.open("rb") as stream:
        dataset_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    start = time.perf_counter()
    p = predict(model, data, np.arange(n), "baseline", args.batch_size)
    m = metrics(p, data["labels"])
    if args.baseline_cache:
        args.baseline_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.baseline_cache,
            dataset_sha256=dataset_sha,
            weights_sha256=model.manifest["files"]["policy.safetensors"],
            **m,
        )
    elapsed = time.perf_counter() - start
    print(f"baseline {n}: {summarize(m)} ({elapsed:.1f}s)", flush=True)
    report = dict(
        schema="shogi-human-policy-evaluation-v1",
        model_version=model.manifest.get("model_version") or model.manifest.get("status", "unknown"),
        weights_sha256=model.manifest["files"]["policy.safetensors"],
        dataset_sha256=dataset_sha,
        split="validation",
        precision="float32 " + args.device.upper() + ("; TF32 disabled" if args.device == "cuda" else ""),
        environment=dict(torch=torch.__version__, numpy=np.__version__),
        normalization="legal moves, temperature=1",
        rating_group_range=dict(lower_inclusive=0, upper_exclusive=3000, width=200),
        baseline=summarize(m),
        runtime=dict(seconds=elapsed, batch_size=args.batch_size, threads=args.threads),
        groups={},
        ablations={},
        interventions={},
    )

    def groups(name, value, boundaries, known=None):
        edges = [-np.inf, *boundaries, np.inf]
        bins = np.digitize(value, boundaries)
        known = np.ones(n, bool) if known is None else known
        entries = []
        for i in range(len(edges) - 1):
            mask = (bins == i) & known
            if mask.any():
                entries.append(
                    dict(
                        lower=None if i == 0 else edges[i],
                        upper=None if i == len(edges) - 2 else edges[i + 1],
                        **summarize(m, mask),
                    )
                )
        report["groups"][name] = entries
        if (~known).any():
            report["groups"][name].append(dict(unknown=True, **summarize(m, ~known)))

    for column, name in enumerate(("self_rating", "opponent_rating")):
        report["groups"][name] = rating_groups(
            data["ratings"][:, column], data["condition_known"][:, 0] > 0, m
        )
    groups("ply", data["game_ply"], [21, 61])
    groups("self_seconds", data["clock_seconds"][:, 0], [10, 30, 60, 300], data["condition_known"][:, 1] > 0)
    rng = np.random.default_rng(20261120)
    diagnostic = np.sort(rng.choice(n, min(n, 5000), replace=False))
    baseline = summarize({k: v[diagnostic] for k, v in m.items()})
    report["diagnostic_baseline"] = baseline
    report["diagnostic_seed"] = 20261120
    for mode in ("no_rating", "no_clock", "neither", "current_only"):
        q = predict(model, data, diagnostic, mode, args.batch_size)
        result = summarize(metrics(q, data["labels"][diagnostic]))
        result["delta_nll"] = result["nll"] - baseline["nll"]
        result["top_choice_changed"] = float((q.argmax(1) != p[diagnostic].argmax(1)).mean())
        report["ablations"][mode] = result
        print(mode, result, flush=True)
    # Controlled low/high rating intervention on the same known-rating positions.
    known = diagnostic[data["condition_known"][diagnostic, 0] > 0]
    intervention = np.sort(rng.choice(known, min(len(known), 2000), replace=False))
    report["intervention_selection"] = (
        "uniform without replacement among diagnostic positions with known ratings; shared RNG after diagnostic draw"
    )
    if len(intervention):
        for side in ("self", "opponent"):
            low = predict(model, data, intervention, side + "_500", args.batch_size)
            high = predict(model, data, intervention, side + "_2500", args.batch_size)
            mid = (low + high) / 2
            tiny = np.finfo(float).tiny
            js = 0.5 * (
                (low * np.log(np.maximum(low, tiny) / np.maximum(mid, tiny))).sum(1)
                + (high * np.log(np.maximum(high, tiny) / np.maximum(mid, tiny))).sum(1)
            )
            report["interventions"][side] = dict(
                n=len(intervention),
                low=500,
                high=2500,
                top_choice_changed=float((low.argmax(1) != high.argmax(1)).mean()),
                js_nats=float(js.mean()),
            )
            print(side, report["interventions"][side], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
