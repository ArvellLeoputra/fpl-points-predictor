"""
Plot the backtest for the README: the total real points of the model squad, the
most-owned squad and the baseline squad (top), and the model minus the
most-owned squad in every gameweek (bottom).
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import ROOT
from src.optimizer.backtest import RESULTS_PATH

OUTPUT_PATH = ROOT / "docs" / "backtest.png"
MODEL_COLOR = "#1f6feb"
POPULAR_COLOR = "#bf8700"
BASELINE_COLOR = "#8b949e"
WIN_COLOR = "#2da44e"
LOSS_COLOR = "#cf222e"
TEXT_GREY = "#57606a"
LABEL_GAP = 0.075  # smallest distance between two end labels, as a share of the highest line


def spread_labels(positions, gap):
    """
    Move the y positions of the end labels apart, so that two lines that end
    close to each other do not get labels on top of each other.
    """
    order = sorted(positions, key=positions.get, reverse=True)
    placed = {}
    for name in order:
        y = positions[name]
        if placed:
            y = min(y, min(placed.values()) - gap)
        placed[name] = y
    return placed


def main():
    """
    Read the saved backtest results and save the chart.
    """
    results = pd.read_csv(RESULTS_PATH, index_col="round")
    total_points = results[["model", "popular", "baseline"]].cumsum()
    rounds = results.index
    last_round = rounds[-1]
    difference = results["model"] - results["popular"]
    wins = (difference > 0).sum()

    plt.rcParams.update({
        "font.size": 10,
        "axes.edgecolor": "#d0d7de",
        "axes.labelcolor": TEXT_GREY,
        "xtick.color": TEXT_GREY,
        "ytick.color": TEXT_GREY,
    })
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.12},
    )

    lines = {
        "baseline": ("Recent form", BASELINE_COLOR, 2),
        "popular": ("Most-owned", POPULAR_COLOR, 2),
        "model": ("Model", MODEL_COLOR, 2.8),
    }
    for key, (label, color, width) in lines.items():
        top.plot(rounds, total_points[key], color=color, linewidth=width)

    # labels at the end of the lines, moved apart if they would overlap
    ends = {key: total_points[key].iloc[-1] for key in lines}
    label_y = spread_labels(ends, LABEL_GAP * max(ends.values()))
    for key, (label, color, width) in lines.items():
        top.annotate(f"{label}  {ends[key]:,.0f} pts", (last_round, ends[key]),
                     xytext=(last_round + 0.8, label_y[key]), textcoords="data",
                     va="center", color=color, fontweight="bold" if key == "model" else "normal",
                     arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=0, shrinkB=2))
    top.set_ylabel("Total points")
    top.set_ylim(0, None)
    top.yaxis.set_major_formatter("{x:,.0f}")

    # bottom: model against the stronger reference, green if the model was ahead
    colors = [WIN_COLOR if d > 0 else LOSS_COLOR for d in difference]
    bottom.bar(rounds, difference, color=colors, width=0.7)
    bottom.axhline(0, color="#d0d7de", linewidth=1)
    bottom.set_ylabel("Model minus\nmost-owned")
    bottom.set_xlabel("Gameweek")
    bottom.text(1.01, 0.5, f"Model ahead\nin {wins} of {len(rounds)}\ngameweeks",
                transform=bottom.transAxes, va="center",
                color=WIN_COLOR, fontweight="bold")

    for ax in (top, bottom):
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    bottom.set_xlim(rounds[0] - 0.5, last_round + 0.5)

    fig.text(0.125, 0.97, "The model scored more points than recent form and the most-owned players",
             fontweight="bold", fontsize=12)
    fig.text(0.125, 0.935,
             "Real points of the starting 11 and the captain (double), one new squad per gameweek, 2025-26 season",
             color=TEXT_GREY, fontsize=9.5)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
