"""Compare self-rating inputs on the same stratified validation positions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from evaluate import FEATURES, metrics, predict, summarize

from shogi_human_policy import PolicyModel

BOUNDARIES = [500, 1000, 1500, 2000, 2500, 3000]
INPUT_RATINGS = [250, 750, 1250, 1500, 1750, 2250, 2750, 3250]
BAND_NAMES = ["500未満", "500〜999", "1000〜1499", "1500〜1999", "2000〜2499", "2500〜2999", "3000以上"]


def stratified_indices(ratings, known, cap, seed):
    rng = np.random.default_rng(seed)
    bands = np.digitize(ratings, BOUNDARIES)
    selected = []
    for band in range(len(BOUNDARIES) + 1):
        candidates = np.flatnonzero((bands == band) & known)
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
    args = parser.parse_args()
    if args.per_band < 1:
        parser.error("--per-band must be positive")
    torch.set_num_threads(args.threads)
    model = PolicyModel.from_directory(args.model)
    with np.load(args.dataset, allow_pickle=False) as archive:
        data = {k: archive[k] for k in (*FEATURES, "legal_policy_mask", "labels")}
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
        precision="float32 CPU",
        environment=dict(torch=torch.__version__, numpy=np.__version__),
        normalization="legal moves, temperature=1",
        seed=args.seed,
        per_band_cap=args.per_band,
        sampling="uniform without replacement within each known self-rating band",
        input_ratings=INPUT_RATINGS,
        fixed="observed opponent rating, clocks, full history",
        bands=[],
        cells=[],
        observed=[],
    )
    selected_bands = bands[indices]
    for band, name in enumerate(BAND_NAMES):
        report["bands"].append(
            dict(
                label=name,
                n=int((selected_bands == band).sum()),
                available=int(((bands == band) & (data["condition_known"][:, 0] > 0)).sum()),
            )
        )
    for rating in [*INPUT_RATINGS, None]:
        mode = "baseline" if rating is None else f"self_{rating}"
        p = predict(model, data, indices, mode, args.batch_size)
        values = metrics(p, data["labels"][indices])
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


if __name__ == "__main__":
    main()
