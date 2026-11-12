"""
Plot the backtest for the README in a football style: the total points of the
model, most-owned and recent-form squads (top), and the model minus the
most-owned squad in every gameweek (bottom).
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from src.config import ROOT
from src.optimizer.backtest import RESULTS_PATH
from src.plots.style import BOLD_FONT, GREEN, SHADE, TEXT_FONT, TITLE_FONT

OUTPUT_PATH = ROOT / "docs" / "backtest.png"
MODEL = GREEN
POPULAR = "#d4a72c"
BASELINE = "#9aa3ad"
LOSS = "#e5534b"
AXIS_GREY = "#8b949e"
STRIPE_ALPHA = 0.08
LABEL_GAP = 0.105
TICKS = [6, 10, 15, 20, 25, 30, 35, 38]


def spread_labels(positions, gap):
    """
    Move the end labels apart, so that lines that end close together do not overlap.
    """
    placed = {}
    for name in sorted(positions, key=positions.get, reverse=True):
        y = positions[name]
        if placed:
            y = min(y, min(placed.values()) - gap)
        placed[name] = y
    return placed


def pitch_stripes(ax, low, high, count=8):
    step = (high - low) / count
    for i in range(0, count, 2):
        ax.axhspan(low + i * step, low + (i + 1) * step, color=MODEL, alpha=STRIPE_ALPHA, zorder=0)


def style_axis(ax):
    """
    No frame except a thin bottom line, and tick labels in Barlow.
    """
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_GREY)
    ax.spines["bottom"].set_alpha(0.5)
    ax.tick_params(left=False, colors=AXIS_GREY, labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(TEXT_FONT)


def main():
    """
    Read the saved backtest results and save the chart with a transparent background.
    """
    results = pd.read_csv(RESULTS_PATH, index_col="round")
    total_points = results[["model", "popular", "baseline"]].cumsum()
    rounds = results.index
    last_round = rounds[-1]
    difference = results["model"] - results["popular"]
    wins = (difference > 0).sum()

    fig = plt.figure(figsize=(10, 6.4))
    top = fig.add_axes([0.075, 0.385, 0.715, 0.455])
    bottom = fig.add_axes([0.075, 0.085, 0.715, 0.2], sharex=top)

    # header band, like the banner
    fig.patches.append(Rectangle((0, 0.89), 1, 0.11, transform=fig.transFigure, color=SHADE, zorder=0))
    fig.text(0.035, 0.945, "Backtest on the 2025-26 season", color="white", fontproperties=TITLE_FONT,
             fontsize=22, va="center")
    fig.text(0.965, 0.945, "one new squad every gameweek  ·  starting 11 + captain", color="#9be9a8",
             fontproperties=TEXT_FONT, fontsize=11.5, va="center", ha="right")

    # top: total points, with the totals as coloured chips at the end of the lines
    high = total_points.values.max() * 1.08
    pitch_stripes(top, 0, high)
    lines = {"baseline": ("Recent form", BASELINE, 2), "popular": ("Most-owned", POPULAR, 2.2),
             "model": ("Model", MODEL, 3.2)}
    for key, (label, color, width) in lines.items():
        top.plot(rounds, total_points[key], color=color, linewidth=width, solid_capstyle="round", zorder=2)
    ends = {key: total_points[key].iloc[-1] for key in lines}
    label_y = spread_labels(ends, LABEL_GAP * high)
    for key, (label, color, width) in lines.items():
        top.annotate(f" {label}  {ends[key]:,.0f} ", (last_round, ends[key]),
                     xytext=(last_round + 1.0, label_y[key]), textcoords="data", va="center",
                     color="white", fontproperties=BOLD_FONT, fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.6", facecolor=color, edgecolor="none"),
                     arrowprops=dict(arrowstyle="-", color=color, lw=1, shrinkA=0, shrinkB=3),
                     annotation_clip=False)
    top.set_ylim(0, high)
    top.set_title("Total points", loc="left", fontproperties=BOLD_FONT, fontsize=12, color=AXIS_GREY, pad=8)
    top.yaxis.set_major_formatter("{x:,.0f}")
    top.tick_params(labelbottom=False, bottom=False)

    # bottom: model minus most-owned in each gameweek, green if the model was ahead
    bottom.axhspan(0, max(difference.max(), 1) * 1.15, color=MODEL, alpha=STRIPE_ALPHA, zorder=0)
    bottom.bar(rounds, difference, color=[MODEL if d > 0 else LOSS for d in difference], width=0.62, zorder=2)
    bottom.axhline(0, color=AXIS_GREY, alpha=0.5, linewidth=1, zorder=1)
    bottom.set_title("Model minus most-owned, per gameweek", loc="left", fontproperties=BOLD_FONT,
                     fontsize=12, color=AXIS_GREY, pad=8)
    bottom.text(1.025, 0.62, f"{wins} of {len(rounds)}", transform=bottom.transAxes, va="center",
                color=MODEL, fontproperties=TITLE_FONT, fontsize=26)
    bottom.text(1.025, 0.25, "gameweeks with the\nmodel ahead", transform=bottom.transAxes, va="center",
                color=AXIS_GREY, fontproperties=TEXT_FONT, fontsize=10.5, linespacing=1.2)

    bottom.set_xticks(TICKS, [f"GW{t}" for t in TICKS])
    bottom.set_xlim(rounds[0] - 0.6, last_round + 0.6)
    for ax in (top, bottom):
        style_axis(ax)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, transparent=True)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
