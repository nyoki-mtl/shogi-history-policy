"""Render Japanese figures from aggregate evaluation JSON."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator


def configure_fonts():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    choices = [
        "Noto Sans CJK JP",
        "IPAexGothic",
        "BIZ UDPGothic",
        "Meiryo",
        "Hiragino Sans",
        "Yu Gothic",
        "BIZ UDGothic",
    ]
    available = [name for name in choices if name in installed]
    if not available:
        raise RuntimeError("Install a Japanese font, such as Noto Sans CJK JP, before rendering figures.")
    plt.rcParams.update(
        {
            "font.family": available[0],
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "shogi-human-policy",
        }
    )


def save(fig, path, *, vector=False):
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)
    if vector:
        svg = path.with_suffix(".svg")
        fig.savefig(svg, bbox_inches="tight", pad_inches=0.18, metadata={"Date": None})
        svg.write_text(
            "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    plt.close(fig)


def dot_panels(entries, labels, title, subtitle, path):
    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, max(3.5, len(entries) * 0.45 + 1.8)), sharey=True, layout="constrained"
    )
    y = np.arange(len(entries))
    labels = [f"{label}　{entry['n']:,}局面" for label, entry in zip(labels, entries, strict=True)]
    for ax, key, name, color in zip(
        axes,
        ["top1", "nll"],
        ["人間の着手との正解率（%）", "着手予測の損失（NLL、低いほど良い）"],
        ["#2166ac", "#b65b29"],
        strict=True,
    ):
        values = np.array([entry[key] * (100 if key == "top1" else 1) for entry in entries])
        span = max(float(np.ptp(values)), 1 if key == "top1" else 0.05)
        ax.scatter(values, y, s=65, color=color, zorder=3)
        for value, row in zip(values, y, strict=True):
            ax.text(value + span * 0.045, row, f"{value:.2f}", va="center", fontsize=11)
        ax.set_xlim(values.min() - span * 0.15, values.max() + span * 0.42)
        ax.set_title(name, loc="left", pad=15)
        ax.grid(axis="x", color="#e3e8ed", zorder=0)
        ax.tick_params(axis="both", length=0, pad=8)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=14, linespacing=1.8)
    save(fig, path)


def plot_standard(report, output, *, rating_entries=None):
    entries = rating_entries
    if entries is None:
        entries = [entry for entry in report["groups"]["self_rating"] if not entry.get("unknown")]
    subtitle = f"{report['snapshot_step']:,}更新 EMA ／ 検証用 {report['baseline']['n']:,}局面 ／ float32"
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 5.6), sharex=True, layout="constrained")
    centers = [(entry["lower"] + entry["upper"]) / 2 for entry in entries]
    for ax, key, title, color in zip(
        axes, ["top1", "nll"], ["正解率（%）", "損失（NLL）"], ["#2166ac", "#b65b29"], strict=True
    ):
        values = [entry[key] * (100 if key == "top1" else 1) for entry in entries]
        ax.plot(centers, values, marker="o", markersize=4, color=color, linewidth=1.6)
        if all(key + "_ci95" in entry for entry in entries):
            ci = np.array([entry[key + "_ci95"] for entry in entries]) * (100 if key == "top1" else 1)
            ax.fill_between(centers, ci[:, 0], ci[:, 1], color=color, alpha=0.12, linewidth=0)
        ax.set_xlim(0, 3000)
        ax.set_xticks(np.arange(0, 3001, 200))
        ax.tick_params(axis="x", labelsize=10, length=0, pad=7)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", pad=12)
        ax.grid(axis="y", color="#e3e8ed")
    axes[-1].set_xlabel("実R（200刻み）", labelpad=10)
    save(fig, output / "rating-performance.png")
    entries = report["groups"]["ply"]
    dot_panels(
        entries,
        ["1〜20手", "21〜60手", "61手以降"],
        "手数帯ごとの予測性能",
        subtitle,
        output / "phase-performance.png",
    )
    modes = ["no_rating", "no_clock", "neither", "current_only"]
    labels = ["Rを省略", "時計を省略", "Rと時計を省略", "過去の局面を省略"]
    deltas = [report["ablations"][mode]["delta_nll"] for mode in modes]
    changes = [100 * report["ablations"][mode]["top_choice_changed"] for mode in modes]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True, layout="constrained")
    y = np.arange(len(modes))
    axes[0].barh(y, deltas, color=["#b65b29" if v >= 0 else "#2166ac" for v in deltas], height=0.52)
    axes[1].barh(y, changes, color="#2166ac", height=0.52)
    for ax, values, title, fmt in zip(
        axes,
        [deltas, changes],
        ["着手予測の損失の増減（NLL）", "第一候補が変わった割合（%）"],
        ["+.4f", ".2f"],
        strict=True,
    ):
        extent = max(abs(value) for value in values)
        ax.set_xlim(min(0, min(values) - extent * 0.2), max(values) + extent * 0.36)
        ax.axvline(0, color="#7b8794", lw=0.8)
        ax.set_title(title, loc="left", pad=15)
        ax.grid(axis="x", color="#e3e8ed", alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=8)
        for row, value in enumerate(values):
            ax.text(value + extent * 0.025, row, format(value, fmt), va="center", fontsize=11)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    axes[0].set_xlabel("負なら改善、正なら悪化", labelpad=10)
    fig.suptitle(
        f"入力を省いたときの変化\n同じ検証用{report['diagnostic_baseline']['n']:,}局面で、入力ありの予測と比較",
        fontsize=14,
        linespacing=1.8,
    )
    save(fig, output / "input-effects.png")


def plot_matrix(report, output):
    if len(report["cells"]) != len(report["input_ratings"]) or not report["observed"]:
        raise ValueError("rating matrix evaluation is incomplete")
    accuracy = np.array(
        [[np.nan if cell is None else cell["top1"] * 100 for cell in col] for col in report["cells"]]
    ).T
    nll = np.array([[np.nan if cell is None else cell["nll"] for cell in col] for col in report["cells"]]).T
    baseline = report["input_ratings"].index(1500)
    versions = [
        ("rating-input-accuracy", accuracy, "着手の正解率", "正解率（%）", "YlGnBu", False, True),
        (
            "rating-input-accuracy-delta",
            accuracy - accuracy[:, baseline, None],
            "正解率の変化（1500固定比）",
            "差（ポイント）",
            "RdBu",
            True,
            True,
        ),
        (
            "rating-input-nll",
            nll - nll[:, baseline, None],
            "損失の変化（1500固定比）",
            "NLL差",
            "RdBu_r",
            True,
            False,
        ),
    ]
    for filename, values, title, label, cmap_name, centered, maximize in versions:
        fig, ax = plt.subplots(figsize=(10.8, 9), layout="constrained")
        if centered:
            limit = max(float(np.nanmax(np.abs(values))), 0.0001)
            low, high = -limit, limit
        else:
            low, high = float(np.nanmin(values)), float(np.nanmax(values))
        picture = ax.imshow(
            values,
            cmap=cmap_name,
            vmin=low,
            vmax=high,
            extent=(0, 3000, 3000, 0),
            aspect="equal",
            interpolation="nearest",
        )
        edges = np.arange(0, 3001, 200)
        ax.set_xticks(edges)
        ax.set_yticks(edges)
        ax.tick_params(length=0, pad=7, labelsize=10)
        ax.set_xlabel("入力R", labelpad=12)
        ax.set_ylabel("実R", labelpad=12)
        ax.set_title(title, loc="left", fontsize=16, pad=16)
        ax.grid(color="white", linewidth=0.45, alpha=0.6)
        for row in range(values.shape[0]):
            if not np.isfinite(values[row]).any():
                continue
            best = np.nanmax(values[row]) if maximize else np.nanmin(values[row])
            columns = np.flatnonzero(np.isclose(values[row], best, rtol=0, atol=1e-10))
            ax.scatter(
                np.asarray(report["input_ratings"])[columns],
                np.full(len(columns), (report["bands"][row]["lower"] + report["bands"][row]["upper"]) / 2),
                s=38,
                facecolor="white",
                edgecolor="#263746",
                linewidth=1,
                zorder=3,
            )
        colorbar = fig.colorbar(picture, ax=ax, fraction=0.035, pad=0.025, shrink=0.78)
        colorbar.locator = MaxNLocator(5)
        colorbar.update_ticks()
        colorbar.set_label(label, labelpad=10)
        save(fig, output / f"{filename}.png", vector=True)


def plot_high_rating_response(report, output):
    """Paired uncertainty against input R2900 in the highest actual rating band."""
    reference = report["input_ratings"].index(2900)
    entries = [column[-1] for column in report["cells"]]
    x = np.array(report["input_ratings"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for ax, key, scale, label, color in zip(
        axes,
        ["top1", "nll"],
        [100, 1],
        ["正解率の差（ポイント）", "NLLの差"],
        ["#2166ac", "#b65b29"],
        strict=True,
    ):
        values = np.array([e[key] - entries[reference][key] for e in entries]) * scale
        ci = np.array([e[key + "_delta_vs_2900_ci95"] for e in entries]) * scale
        ax.axhline(0, color="#7b8794", lw=0.9)
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.15, linewidth=0)
        ax.plot(x, values, color=color, marker="o", markersize=3, linewidth=1.7)
        ax.set_xlim(0, 3000)
        ax.set_xticks(np.arange(0, 3001, 600))
        ax.set_xlabel("自分の入力R")
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#e3e8ed", alpha=0.7)
        ax.tick_params(length=0, pad=7)
    axes[0].set_title("正解率（上ほど良い）", loc="left")
    axes[1].set_title("損失（下ほど良い）", loc="left")
    save(fig, output / "high-rating-input-response.png", vector=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--rating-matrix", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    configure_fonts()
    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    with args.metrics.with_name(args.metrics.stem + "-groups.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["group", "lower", "upper", "unknown", "n", "nll", "top1", "entropy"],
            lineterminator="\n",
        )
        writer.writeheader()
        for group, entries in report["groups"].items():
            writer.writerows(dict(group=group, **entry) for entry in entries)
    matrix = None
    rating_entries = None
    if args.rating_matrix:
        matrix = json.loads(args.rating_matrix.read_text(encoding="utf-8"))
        if matrix["weights_sha256"] != report["weights_sha256"] or matrix["split"] != report["split"]:
            raise ValueError("matrix model or evaluation split differs from standard report")
        if matrix.get("matched_selection"):
            rating_entries = [
                dict(lower=band["lower"], upper=band["upper"], **entry)
                for band, entry in zip(matrix["bands"], matrix["observed"], strict=True)
            ]
    plot_standard(report, args.output, rating_entries=rating_entries)
    if matrix:
        plot_matrix(matrix, args.output)
        if matrix.get("matched_selection"):
            plot_high_rating_response(matrix, args.output)


if __name__ == "__main__":
    main()
