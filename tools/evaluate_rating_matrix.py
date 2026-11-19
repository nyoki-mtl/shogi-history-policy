"""Compare self-rating inputs on the same stratified validation positions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from evaluate import FEATURES, metrics, predict, summarize
from evaluation_stats import game_bootstrap

from shogi_human_policy import PolicyModel

BOUNDARIES = list(range(200, 3000, 200))
INPUT_RATINGS = list(range(100, 3000, 200))
BAND_NAMES = [f"{lower}〜{lower + 199}" for lower in range(0, 3000, 200)]


def stratified_indices(ratings, known, cap, seed):
    rng = np.random.default_rng(seed)
    bands = np.digitize(ratings, BOUNDARIES)
    selected = []
    for band in range(len(BOUNDARIES) + 1):
        candidates = np.flatnonzero((bands == band) & known & (ratings >= 0) & (ratings < 3000))
        selected.extend(rng.choice(candidates, min(len(candidates), cap), replace=False))
    return np.sort(selected).astype(np.int64), bands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-band", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--cache", type=Path, help="Private per-position metrics, resumable across rating inputs"
    )
    parser.add_argument(
        "--selection-info", type=Path, help="Aggregate provenance of the matched validation sample"
    )
    args = parser.parse_args()
    if args.per_band < 1:
        parser.error("--per-band must be positive")
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(0.15)
    model = PolicyModel.from_directory(args.model, device=args.device)
    with np.load(args.dataset, allow_pickle=False) as archive:
        keys = [*FEATURES, "legal_policy_mask", "labels"]
        if args.selection_info:
            keys.append("game_id")
        data = {k: archive[k] for k in keys}
    indices, bands = stratified_indices(
        data["ratings"][:, 0], data["condition_known"][:, 0] > 0, args.per_band, args.seed
    )
    with args.dataset.open("rb") as stream:
        dataset_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(
        schema="shogi-human-policy-rating-matrix-v1",
        snapshot_step=model.manifest["step"],
        weights_sha256=model.manifest["files"]["policy.safetensors"],
        dataset_sha256=dataset_sha,
        split="validation",
        precision="float32 " + args.device.upper() + ("; TF32 disabled" if args.device == "cuda" else ""),
        environment=dict(torch=torch.__version__, numpy=np.__version__),
        normalization="legal moves, temperature=1",
        seed=args.seed,
        per_band_cap=args.per_band,
        sampling="uniform without replacement within each known self-rating band; 0 <= rating < 3000",
        input_ratings=INPUT_RATINGS,
        fixed="observed opponent rating, clocks, full history",
        bands=[],
        cells=[],
        observed=[],
    )
    if args.selection_info:
        selection = json.loads(args.selection_info.read_text(encoding="utf-8"))
        if selection["sha256"] != dataset_sha or selection["split"] != "validation":
            raise ValueError("Selection provenance differs from dataset")
        if args.per_band != selection["per_band"]:
            raise ValueError("Matched sample must be evaluated in full")
        report["matched_selection"] = selection
        report["sampling"] = selection["selection"]
        report["bootstrap"] = dict(
            unit="game",
            repetitions=2000,
            seed=20261120,
            confidence=0.95,
            intervals="pointwise; paired differences versus input R1500",
        )
    selected_bands = bands[indices]
    for band, name in enumerate(BAND_NAMES):
        report["bands"].append(
            dict(
                label=name,
                lower=200 * band,
                upper=200 * (band + 1),
                n=int((selected_bands == band).sum()),
                available=int(
                    (
                        (bands == band)
                        & (data["condition_known"][:, 0] > 0)
                        & (data["ratings"][:, 0] >= 0)
                        & (data["ratings"][:, 0] < 3000)
                    ).sum()
                ),
            )
        )
    cached = {}
    if args.cache and args.cache.exists():
        with np.load(args.cache, allow_pickle=False) as archive:
            if (
                str(archive["dataset_sha256"]) != dataset_sha
                or str(archive["weights_sha256"]) != report["weights_sha256"]
                or str(archive["precision"]) != report["precision"]
                or not np.array_equal(archive["indices"], indices)
            ):
                raise ValueError("Private metric cache identity differs")
            cached = {k: archive[k] for k in archive.files if k.startswith("metrics_")}
    values_by_input = []
    for rating in [*INPUT_RATINGS, None]:
        mode = "baseline" if rating is None else f"self_{rating}"
        cache_key = "metrics_" + mode
        if cache_key in cached:
            values = dict(zip(("nll", "top1", "entropy"), cached[cache_key], strict=True))
        else:
            p = predict(model, data, indices, mode, args.batch_size)
            values = metrics(p, data["labels"][indices])
            cached[cache_key] = np.stack([values[k] for k in ("nll", "top1", "entropy")])
            if args.cache:
                args.cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.cache,
                    dataset_sha256=dataset_sha,
                    weights_sha256=report["weights_sha256"],
                    precision=report["precision"],
                    indices=indices,
                    **cached,
                )
        values_by_input.append(values)
        summaries = [
            summarize(values, selected_bands == band) if (selected_bands == band).any() else None
            for band in range(len(BAND_NAMES))
        ]
        if rating is None:
            report["observed"] = summaries
        else:
            report["cells"].append(summaries)
        print(
            "observed" if rating is None else rating,
            [None if s is None else round(s["top1"] * 100, 2) for s in summaries],
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    if args.selection_info:
        for band in range(len(BAND_NAMES)):
            mask = selected_bands == band
            for metric in ("top1", "nll"):
                v = np.stack([x[metric][mask] for x in values_by_input])
                intervals = game_bootstrap(v, data["game_id"][indices][mask], reference=7)
                entries = [column[band] for column in report["cells"]] + [report["observed"][band]]
                for i, entry in enumerate(entries):
                    entry[metric + "_ci95"] = intervals["mean_ci95"][i]
                    entry[metric + "_delta_vs_1500_ci95"] = intervals["delta_ci95"][i]
                if band == len(BAND_NAMES) - 1:
                    high_intervals = game_bootstrap(v, data["game_id"][indices][mask], reference=14)
                    for i, entry in enumerate(entries):
                        entry[metric + "_delta_vs_2900_ci95"] = high_intervals["delta_ci95"][i]
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )


if __name__ == "__main__":
    main()
