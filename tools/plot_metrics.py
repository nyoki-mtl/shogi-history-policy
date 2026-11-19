"""Render compact figures from aggregate evaluation JSON only."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    with args.metrics.with_name(args.metrics.stem + "-groups.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["group", "lower", "upper", "unknown", "n", "nll", "top1", "entropy"]
        )
        writer.writeheader()
        for group, entries in report["groups"].items():
            writer.writerows(dict(group=group, **entry) for entry in entries)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    blue, red = "#2563a6", "#c26b31"
    entries = [x for x in report["groups"]["self_rating"] if not x.get("unknown")]
    labels = [
        f"<{x['upper']}"
        if x["lower"] is None
        else f"{x['lower']}+"
        if x["upper"] is None
        else f"{x['lower']}–{x['upper'] - 1}"
        for x in entries
    ]
    labels = [f"{label}\n(n={entry['n']:,})" for label, entry in zip(labels, entries, strict=True)]
    x = np.arange(len(entries))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, key, title, color in zip(
        axes,
        ["top1", "nll"],
        ["Human-move accuracy (%)", "Policy NLL (lower is better)"],
        [blue, red],
        strict=True,
    ):
        values = [e[key] * (100 if key == "top1" else 1) for e in entries]
        ax.plot(x, values, marker="o", color=color, lw=2)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Player rating")
        ax.grid(axis="y", alpha=0.18)
        for i, v in enumerate(values):
            ax.annotate(
                f"{v:.1f}" if key == "top1" else f"{v:.2f}",
                (i, v),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.25)
    fig.suptitle(
        f"Step {report['snapshot_step']:,} EMA · validation {report['baseline']['n']:,} positions · float32",
        fontsize=12,
    )
    fig.savefig(args.output / "rating-performance.png", dpi=180)
    plt.close(fig)

    modes = ["no_rating", "no_clock", "neither", "current_only"]
    names = ["Without ratings", "Without clocks", "Without both", "Current board only"]
    values = [report["ablations"][m]["delta_nll"] for m in modes]
    changes = [100 * report["ablations"][m]["top_choice_changed"] for m in modes]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for ax, vals, title, color, fmt in zip(
        axes,
        [values, changes],
        ["Increase in policy NLL", "Top choice changed (%)"],
        [red, blue],
        ["+.4f", ".1f"],
        strict=True,
    ):
        ax.barh(names[::-1], vals[::-1], color=color, height=0.6)
        ax.axvline(0, color="#777", lw=0.6)
        ax.set_title(title, loc="left", weight="bold")
        ax.grid(axis="x", alpha=0.18)
        limit = max(abs(v) for v in vals)
        ax.set_xlim(min(0, min(vals) - limit * 0.15), max(vals) + limit * 0.3 + 0.001)
        for y, v in enumerate(vals[::-1]):
            ax.text(v + limit * 0.025, y, format(v, fmt), va="center", fontsize=10)
    fig.suptitle(
        f"Same {report['diagnostic_baseline']['n']:,} validation positions · full inputs as reference",
        fontsize=12,
    )
    fig.savefig(args.output / "input-effects.png", dpi=180)
    plt.close(fig)

    entries = report["groups"]["ply"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    labels = ["Ply 1–20", "Ply 21–60", "Ply 61+"]
    for ax, key, title, color in zip(
        axes, ["top1", "nll"], ["Human-move accuracy (%)", "Policy NLL"], [blue, red], strict=True
    ):
        v = [e[key] * (100 if key == "top1" else 1) for e in entries]
        ax.bar(labels, v, color=color, width=0.55)
        ax.set_ylim(0, max(v) * 1.22)
        ax.set_title(title, loc="left", weight="bold")
        for i, value in enumerate(v):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")
    fig.suptitle(
        f"By ply count · validation {report['baseline']['n']:,} positions · step {report['snapshot_step']:,}",
        fontsize=12,
    )
    fig.savefig(args.output / "phase-performance.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
