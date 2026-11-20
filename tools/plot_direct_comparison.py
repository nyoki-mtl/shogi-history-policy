"""Japanese figures for the paired public Maia2-shogi comparison."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_metrics import configure_fonts, save

MODELS = ("maia2", "human_policy", "human_policy_reduced")
LABELS = ("maia2-shogi", "新モデル", "新モデル（履歴1・時計なし）")
COLORS = ("#7e8b97", "#2166ac", "#d78b4a")


def setup(ax, xlabel):
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", color="#e3e8ed")
    ax.tick_params(length=0, pad=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--rating-matrix", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rating_matrix = (
        json.loads(args.rating_matrix.read_text(encoding="utf-8")) if args.rating_matrix is not None else None
    )
    configure_fonts()
    args.output.mkdir(parents=True, exist_ok=True)
    x = np.array([(b["lower"] + b["upper"]) / 2 for b in report["bands"]])
    rm_x = (
        np.array([(b["lower"] + b["upper"]) / 2 for b in rating_matrix["bands"]])
        if rating_matrix is not None
        else None
    )
    for group, prefix, xlabel in (
        ("bands", "direct-maia-rating", "実R（200刻み）"),
        ("phases", "direct-maia-phase", "初期局面からの着手数"),
    ):
        values = report[group]
        coords = x if group == "bands" else np.arange(4)
        for key, scale, title, suffix in (
            ("top1", 100, "着手の正解率（%）", "accuracy"),
            ("nll", 1, "損失（NLL、小さいほど良い）", "nll"),
        ):
            fig, ax = plt.subplots(figsize=(7.6, 4.2), layout="constrained")
            for model, label, color in zip(MODELS, LABELS, COLORS, strict=True):
                ax.plot(
                    coords,
                    [b["models"][model][key] * scale for b in values],
                    color=color,
                    marker="o",
                    markersize=4,
                    label=label,
                )
            if group == "bands" and rating_matrix is not None:
                ax.plot(
                    rm_x,
                    [o[key] * scale for o in rating_matrix["observed"]],
                    color="#2166ac",
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.75,
                    marker="o",
                    markersize=3,
                    label="新モデル（全R帯 30,000局面・参考）",
                )
            if group == "bands":
                ax.set_xlim(0, 3000)
                ax.set_xticks(np.arange(0, 3001, 200), np.arange(0, 3001, 200), rotation=45, fontsize=9)
            else:
                ax.set_xticks(coords, [b["label"] for b in values], fontsize=10)
            ax.set_title(title, loc="left")
            setup(ax, xlabel)
            ncol = 2 if (group == "bands" and rating_matrix is not None) else 3
            fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=ncol, frameon=False)
            save(fig, args.output / f"{prefix}-{suffix}.png", vector=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey="row", layout="constrained")
    for column, (model, title) in enumerate(zip(MODELS[:2], LABELS[:2], strict=True)):
        for row, key, scale in ((0, "top1", 100), (1, "nll", 1)):
            ax = axes[row, column]
            for rating, color in zip((900, 1600, 2100), ("#d78b4a", "#7e8b97", "#2166ac"), strict=True):
                ax.plot(
                    x,
                    [b["response"][model]["cases"][str(rating)][key] * scale for b in report["bands"]],
                    color=color,
                    marker="o",
                    markersize=3,
                    label=f"入力R{rating}",
                )
            ax.set_xlim(800, 2800)
            ax.set_xticks(np.arange(800, 2801, 200), np.arange(800, 2801, 200), rotation=45, fontsize=9)
            setup(ax, "実R（200刻み）" if row else "")
            if not row:
                ax.set_title(title, loc="left")
    axes[0, 0].set_ylabel("正解率（%）")
    axes[1, 0].set_ylabel("NLL")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
    save(fig, args.output / "direct-maia-rating-response.png", vector=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, key, scale, title in zip(
        axes,
        ("changed", "full_jsd"),
        (100, 1),
        ("R900と2100の第一候補変更率（%）", "R900と2100の着手別JSD（底2）"),
        strict=True,
    ):
        for model, label, color in zip(MODELS[:2], LABELS[:2], COLORS[:2], strict=True):
            ax.plot(
                np.arange(4),
                [b["response"][model]["pairs"]["900_2100"][key] * scale for b in report["phases"]],
                color=color,
                marker="o",
                label=label,
            )
        ax.set_xticks(np.arange(4), [b["label"] for b in report["phases"]], fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left")
        setup(ax, "初期局面からの着手数")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    save(fig, args.output / "direct-maia-response-by-phase.png", vector=True)


if __name__ == "__main__":
    main()
