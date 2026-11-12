"""
Draw the output of the next-gameweek squad as a terminal window, for the README.
The text is the real output of `python -m src.optimizer.next_gameweek`.
"""

import contextlib
import io

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from src.config import ROOT
from src.optimizer import next_gameweek

OUTPUT_PATH = ROOT / "docs" / "squad.png"
COMMAND = "python -m src.optimizer.next_gameweek"
MONO_FONT = FontProperties(family="DejaVu Sans Mono")
MONO_BOLD = FontProperties(family="DejaVu Sans Mono", weight="bold")

WINDOW = "#0d1117"
TITLE_BAR = "#161b22"
BORDER = "#30363d"
TEXT = "#c9d1d9"
PROMPT = "#3fb950"
HEADER = "#58a6ff"
CAPTAIN = "#d29922"
BENCH = "#8b949e"
WHITE = "#ffffff"

LINE_HEIGHT = 0.26
FONT_SIZE = 11


def run_next_gameweek():
    """
    Run the next-gameweek script and return what it prints, line by line.
    """
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        next_gameweek.main()
    return printed.getvalue().rstrip("\n").split("\n")


def line_style(line):
    if "web_name" in line:
        return HEADER, MONO_FONT
    if line.rstrip().endswith("captain"):
        return CAPTAIN, MONO_BOLD
    if line.rstrip().endswith("bench"):
        return BENCH, MONO_FONT
    if line.startswith(("Squad for", "Price", "Predicted")):
        return WHITE, MONO_BOLD
    return TEXT, MONO_FONT


def draw_terminal(output_lines):
    width = 7.4
    bar_height, padding = 0.42, 0.3
    height = bar_height + 2 * padding + (len(output_lines) + 1) * LINE_HEIGHT
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    # window with a title bar and the three coloured buttons
    ax.add_patch(FancyBboxPatch((0.02, 0.02), width - 0.04, height - 0.04, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=WINDOW, edgecolor=BORDER, linewidth=1))
    ax.add_patch(FancyBboxPatch((0.02, height - bar_height - 0.02), width - 0.04, bar_height,
                                boxstyle="round,pad=0,rounding_size=0.12", facecolor=TITLE_BAR, edgecolor="none"))
    ax.add_patch(Rectangle((0.02, height - bar_height - 0.02), width - 0.04, 0.12, facecolor=TITLE_BAR, edgecolor="none"))
    for i, color in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        ax.add_patch(Circle((0.3 + i * 0.25, height - bar_height / 2 - 0.02), 0.075, color=color))
    ax.text(width / 2, height - bar_height / 2 - 0.02, "fpl-points-predictor", color=BENCH,
            fontproperties=MONO_FONT, fontsize=10, ha="center", va="center")

    # the command, then the output
    y = height - bar_height - padding - LINE_HEIGHT / 2
    ax.text(0.35, y, "$", color=PROMPT, fontproperties=MONO_FONT, fontsize=FONT_SIZE, va="center")
    ax.text(0.6, y, COMMAND, color=TEXT, fontproperties=MONO_FONT, fontsize=FONT_SIZE, va="center")
    for index, line in enumerate(output_lines, start=1):
        color, font = line_style(line)
        ax.text(0.35, y - index * LINE_HEIGHT, line, color=color, fontproperties=font, fontsize=FONT_SIZE, va="center")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=170, transparent=True)
    print(f"Saved {OUTPUT_PATH}")


def main():
    draw_terminal(run_next_gameweek())


if __name__ == "__main__":
    main()
