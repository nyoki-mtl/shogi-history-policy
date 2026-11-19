"""Render Japanese figures from aggregate evaluation JSON."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patheffects
from matplotlib.patches import Rectangle


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
        }
    )


def save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def band_label(entry):
    if entry["lower"] is None:
        return f"{entry['upper']}未満"
    if entry["upper"] is None:
        return f"{entry['lower']}以上"
    return f"{entry['lower']}〜{entry['upper'] - 1}"


def dot_panels(entries, labels, title, subtitle, path):
    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, max(3.5, len(entries) * 0.65 + 1.5)), sharey=True, layout="constrained"
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


def plot_standard(report, output):
    entries = [entry for entry in report["groups"]["self_rating"] if not entry.get("unknown")]
    subtitle = f"{report['snapshot_step']:,}更新 EMA ／ 検証用 {report['baseline']['n']:,}局面 ／ float32"
    dot_panels(
        entries,
        [band_label(entry) for entry in entries],
        "実際の棋力帯ごとの予測性能",
        subtitle,
        output / "rating-performance.png",
    )
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
        ("rating-input-accuracy", accuracy, "入力する自分のRと正解率", "正解率（%）", "YlGnBu", False, True),
        (
            "rating-input-accuracy-delta",
            accuracy - accuracy[:, baseline, None],
            "入力Rを1500に固定した場合からの正解率の変化",
            "正解率の差（ポイント）",
            "RdBu",
            True,
            True,
        ),
        (
            "rating-input-nll",
            nll - nll[:, baseline, None],
            "入力Rを1500に固定した場合からの損失の変化",
            "NLLの差（負なら改善）",
            "RdBu_r",
            True,
            False,
        ),
    ]
    for filename, values, title, legend, cmap_name, centered, maximize in versions:
        fig, ax = plt.subplots(figsize=(12.5, 6.5), layout="constrained")
        cmap = plt.get_cmap(cmap_name)
        if centered:
            limit = max(float(np.nanmax(np.abs(values))), 0.0001)
            low, high = -limit, limit
        else:
            low, high = float(np.nanmin(values)), float(np.nanmax(values))
        picture = ax.imshow(values, cmap=cmap, vmin=low, vmax=high, aspect="auto")
        ax.set_xticks(
            np.arange(len(report["input_ratings"])),
            [f"{rating:,}" + ("\n基準" if rating == 1500 else "") for rating in report["input_ratings"]],
        )
        ax.set_yticks(
            np.arange(len(report["bands"])),
            [f"{band['label']}\n{band['n']:,}局面" for band in report["bands"]],
        )
        ax.set_xlabel("モデルへ与える自分のレーティング", labelpad=12)
        ax.set_ylabel("実際に指した人のレーティング帯", labelpad=12)
        ax.tick_params(length=0, pad=9)
        ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="minor", length=0)
        for row in range(values.shape[0]):
            if not np.isfinite(values[row]).any():
                continue
            best = np.nanmax(values[row]) if maximize else np.nanmin(values[row])
            for col, value in enumerate(values[row]):
                if not np.isfinite(value):
                    continue
                is_best = bool(np.isclose(value, best, rtol=0, atol=1e-10))
                rgba = cmap((value - low) / max(high - low, 1e-9))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                label = (
                    f"{value:+.2f}"
                    if centered and maximize
                    else f"{value:+.3f}"
                    if centered
                    else f"{value:.2f}"
                )
                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color="white" if luminance < 0.5 else "#14222e",
                    fontsize=12,
                    weight="bold" if is_best else "normal",
                )
                if is_best:
                    ax.add_patch(
                        Rectangle(
                            (col - 0.46, row - 0.46),
                            0.92,
                            0.92,
                            fill=False,
                            edgecolor="#111111",
                            lw=2.4,
                            path_effects=[
                                patheffects.Stroke(linewidth=4.5, foreground="white"),
                                patheffects.Normal(),
                            ],
                        )
                    )
        colorbar = fig.colorbar(picture, ax=ax, shrink=0.85, pad=0.03)
        colorbar.set_label(legend, labelpad=12)
        ax.set_title(
            f"{title}\n{report['snapshot_step']:,}更新EMA ／ 枠＝行の最良値 ／ 他の入力は固定",
            loc="left",
            fontsize=15,
            pad=20,
            linespacing=1.8,
        )
        save(fig, output / f"{filename}.png")


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
    plot_standard(report, args.output)
    if args.rating_matrix:
        matrix = json.loads(args.rating_matrix.read_text(encoding="utf-8"))
        if (
            matrix["weights_sha256"] != report["weights_sha256"]
            or matrix["dataset_sha256"] != report["dataset_sha256"]
        ):
            raise ValueError("matrix model or evaluation dataset differs from standard report")
        plot_matrix(matrix, args.output)


if __name__ == "__main__":
    main()
