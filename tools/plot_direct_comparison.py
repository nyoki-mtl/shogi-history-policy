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
    parser.add_argument("--paper-diagnostics", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    paper = (
        json.loads(args.paper_diagnostics.read_text(encoding="utf-8"))
        if args.paper_diagnostics is not None
        else None
    )
    configure_fonts()
    args.output.mkdir(parents=True, exist_ok=True)
    x = np.array([(b["lower"] + b["upper"]) / 2 for b in report["bands"]])
    full_x = np.arange(100, 3000, 200)
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
            if group == "bands" and paper is not None:
                ax.plot(
                    coords,
                    [b["models"]["maia2"][key] * scale for b in values],
                    color=COLORS[0],
                    marker="o",
                    markersize=4,
                    label="maia2-shogi",
                )
                for ctx, label, color in (
                    ("observed", "新モデル", COLORS[1]),
                    ("observed_current_only", "新モデル（履歴1のみ）", COLORS[2]),
                ):
                    bands = paper["contexts"][ctx]["bands"]
                    ax.plot(
                        full_x,
                        [b["cases"]["None"][key] * scale for b in bands],
                        color=color,
                        marker="o",
                        markersize=4,
                        label=label,
                    )
            else:
                for model, label, color in zip(MODELS, LABELS, COLORS, strict=True):
                    ax.plot(
                        coords,
                        [b["models"][model][key] * scale for b in values],
                        color=color,
                        marker="o",
                        markersize=4,
                        label=label,
                    )
            if group == "bands":
                ax.set_xlim(0, 3000)
                ax.set_xticks(np.arange(0, 3001, 200), np.arange(0, 3001, 200), rotation=45, fontsize=9)
            else:
                ax.set_xticks(coords, [b["label"] for b in values], fontsize=10)
            ax.set_title(title, loc="left")
            setup(ax, xlabel)
            fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
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

    by_lower = {b["lower"]: b for b in report["bands"]}
    rating_groups = (
        ("低 (800〜1399)", (800, 1000, 1200)),
        ("中 (1400〜1999)", (1400, 1600, 1800)),
        ("高 (2000〜2799)", (2000, 2200, 2400, 2600)),
    )
    jsd_pairs = (
        ("900_1600", "低–中\n(900 vs 1600)"),
        ("1600_2100", "中–高\n(1600 vs 2100)"),
        ("900_2100", "低–高\n(900 vs 2100)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2), sharex=True, sharey=True, layout="constrained")
    for row, (metric_key, metric_label) in enumerate(
        (("rank_jsd", "順位別 Top-5 平均 JSD"), ("full_jsd", "着手別 平均 JSD"))
    ):
        grids = {}
        for model in MODELS[:2]:
            grids[model] = np.array(
                [
                    [
                        np.mean(
                            [by_lower[low]["response"][model]["pairs"][pair][metric_key] for low in lowers]
                        )
                        for pair, _ in jsd_pairs
                    ]
                    for _, lowers in rating_groups
                ]
            )
        vmax = max(float(g.max()) for g in grids.values()) * 1.05
        for col, (model, model_label) in enumerate(zip(MODELS[:2], LABELS[:2], strict=True)):
            ax = axes[row, col]
            grid = grids[model]
            ax.imshow(grid, cmap="Blues", vmin=0.0, vmax=vmax, aspect="auto")
            for i in range(len(rating_groups)):
                for j in range(len(jsd_pairs)):
                    val = grid[i, j]
                    text_color = "#ffffff" if val > vmax * 0.55 else "#1f2937"
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=text_color, fontsize=10.5)
            ax.set_xticks(np.arange(len(jsd_pairs)), [lbl for _, lbl in jsd_pairs], fontsize=9.5)
            ax.set_yticks(np.arange(len(rating_groups)), [lbl for lbl, _ in rating_groups], fontsize=9.5)
            ax.tick_params(length=0, pad=6)
            if row == 0:
                ax.set_title(model_label, loc="left", fontsize=11)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=10)
    save(fig, args.output / "direct-maia-jsd-tables.png", vector=True)


if __name__ == "__main__":
    main()
