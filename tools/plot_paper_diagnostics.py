"""Japanese, chart-first figures for the paper-aligned diagnostic aggregates."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_metrics import configure_fonts, save

COLORS = ("#c87b40", "#6b8295", "#2166ac")
CONTEXT_LABELS = ("8履歴・相手R観測値", "8履歴・相手R1600固定", "現在局面だけ・相手R観測値")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    configure_fonts()
    contexts = report["contexts"]
    full = contexts["full"]
    centers = np.arange(100, 3000, 200)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True, layout="constrained")
    for p, ax in enumerate(axes.flat):
        for rating, color in zip(report["ratings"], COLORS, strict=True):
            y = [b[p]["cases"][str(rating)]["top1"] * 100 for b in full["band_phases"]]
            ax.plot(centers, y, color=color, marker="o", markersize=3, label=f"入力R{rating}")
        ax.set_title(report["phases"][p], loc="left")
        ax.set_xlim(0, 3000)
        ax.set_xticks(np.arange(0, 3001, 600))
        ax.grid(axis="y", color="#e3e8ed")
        ax.tick_params(length=0)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        frameon=False,
        fontsize=10,
        loc="outside upper center",
        ncol=3,
    )
    for ax in axes[:, 0]:
        ax.set_ylabel("着手の正解率（%）")
    for ax in axes[-1]:
        ax.set_xlabel("実R（200刻み）")
    save(fig, args.output / "paper-phase-accuracy.png", vector=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    bottom = np.zeros(15)
    group_values = [
        np.array([b["patterns"][7] for b in full["bands"]]) * 100,
        np.array([b["partial_correct"] for b in full["bands"]]) * 100,
        np.array([b["patterns"][0] for b in full["bands"]]) * 100,
    ]
    for values, label, color in zip(
        group_values,
        ["3条件とも正解", "一部だけ正解", "3条件とも不正解"],
        ["#9ecae1", "#df935b", "#dce1e6"],
        strict=True,
    ):
        axes[0].bar(centers, values, width=175, bottom=bottom, color=color, label=label)
        bottom += values
    axes[0].set_ylim(0, 100)
    axes[0].set_title("人間の手を当てた条件の組合せ", loc="left")
    axes[0].set_ylabel("局面の割合（%）")
    axes[0].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    for context, label, color in zip(
        ("full", "opponent_fixed", "current_only"), CONTEXT_LABELS, COLORS[::-1], strict=True
    ):
        y = [b["pairs"]["900_2100"]["changed"] * 100 for b in contexts[context]["bands"]]
        axes[1].plot(centers, y, color=color, marker="o", markersize=3, label=label)
    axes[1].set_title("R900と2100で第一候補が変わる割合", loc="left")
    axes[1].set_ylabel("第一候補変更率（%）")
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    for ax in axes:
        ax.set_xlim(0, 3000)
        ax.set_xticks(np.arange(0, 3001, 600))
        ax.set_xlabel("実R（200刻み）")
        ax.tick_params(length=0)
    save(fig, args.output / "paper-correctness-patterns.png", vector=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    x = np.arange(4)
    for key, label, color in zip(
        ("rank_jsd", "full_jsd"), ("上位5手の順位別", "全合法手の着手別"), ("#c87b40", "#2166ac"), strict=True
    ):
        y = [p["pairs"]["900_2100"][key] for p in full["phases"]]
        axes[0].plot(x, y, color=color, marker="o", label=label)
    axes[0].set_title("確率の形と、指し手の違い", loc="left")
    axes[0].set_ylabel("JSD（底2、小さいほど近い）")
    axes[0].legend(frameon=False, fontsize=9)
    for context, label, color in zip(
        ("full", "opponent_fixed", "current_only"), CONTEXT_LABELS, COLORS[::-1], strict=True
    ):
        y = [p["pairs"]["900_2100"]["full_jsd"] for p in contexts[context]["phases"]]
        axes[1].plot(x, y, color=color, marker="o", label=label)
    axes[1].set_title("履歴と相手Rによる変化", loc="left")
    axes[1].set_ylabel("全合法手の着手別JSD")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(x, report["phases"], fontsize=10)
        ax.set_ylim(bottom=0)
        ax.tick_params(length=0)
        ax.grid(axis="y", color="#e3e8ed")
    save(fig, args.output / "paper-rank-vs-move-jsd.png", vector=True)

    observed = contexts["observed"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    axes[0].plot([0, 1], [0, 1], color="#aeb8c2", linestyle="--", lw=1)
    bins = observed["overall"]["calibration"]["bins"]
    axes[0].plot(
        [e["confidence"] for e in bins],
        [e["accuracy"] for e in bins],
        color="#2166ac",
        marker="o",
        label="観測R・時計・8履歴",
    )
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="第一候補の予測確率", ylabel="実際の正解率")
    axes[0].set_title("予測確率と正解率の対応", loc="left")
    axes[0].legend(frameon=False, fontsize=9)
    twin = axes[0].twinx()
    twin.bar(
        [e["confidence"] for e in bins],
        [e["n"] / report["n"] * 100 for e in bins],
        width=0.065,
        color="#b8c9d8",
        alpha=0.22,
    )
    twin.set_ylabel("局面の割合（%）", color="#7d8c99")
    twin.set_ylim(0, 40)
    twin.spines["right"].set_visible(False)
    ece = np.array([[p["cases"]["None"]["ece"] * 100 for p in b] for b in observed["band_phases"]])
    pic = axes[1].imshow(
        ece, cmap="YlOrRd", aspect="auto", origin="lower", extent=(-0.5, 3.5, 0, 3000), vmin=0
    )
    axes[1].set_yticks(np.arange(0, 3001, 200))
    axes[1].set_xticks(np.arange(4), report["phases"], fontsize=9)
    axes[1].set_ylabel("実R")
    axes[1].set_title("R帯と手数別の校正誤差", loc="left")
    fig.colorbar(pic, ax=axes[1], fraction=0.04, label="ECE（ポイント）")
    save(fig, args.output / "paper-calibration.png", vector=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for context, label, color in zip(
        ("observed_current_only", "observed_opponent_fixed"),
        ("過去局面を省略", "相手Rを1600固定"),
        ("#c87b40", "#2166ac"),
        strict=True,
    ):
        for ax, key, scale, title in zip(
            axes,
            ("nll", "top1"),
            (1, 100),
            ("NLLの差（下ほど改善）", "正解率の差（上ほど改善）"),
            strict=True,
        ):
            values = [
                (b["cases"]["None"][key] - ref["cases"]["None"][key]) * scale
                for b, ref in zip(contexts[context]["bands"], observed["bands"], strict=True)
            ]
            ax.plot(centers, values, marker="o", markersize=3, label=label, color=color)
            ax.set_title(title, loc="left")
            ax.set_xlim(0, 3000)
            ax.set_xticks(np.arange(0, 3001, 600))
            ax.set_xlabel("実R（200刻み）")
    for ax in axes:
        ax.axhline(0, color="#8b98a5", lw=0.9)
        ax.tick_params(length=0)
        ax.grid(axis="y", color="#e3e8ed")
    axes[0].set_ylabel("NLL差")
    axes[1].set_ylabel("差（ポイント）")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    save(fig, args.output / "paper-actual-input-controls.png", vector=True)


if __name__ == "__main__":
    main()
