"""
Draw the steps of the project for the README: one card per step, like the
player cards of a team sheet, with arrows between them.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.config import ROOT
from src.plots.style import GOLD, GREEN, LINE_GREY, SHADE, TEXT_FONT, TEXT_GREY, TITLE_FONT

OUTPUT_PATH = ROOT / "docs" / "pipeline.png"
ARROW_GREY = "#8c959f"
STEPS = [  # title, description
    ("Data", "FPL API + archive\n5 seasons in SQLite"),
    ("Features", "form of the last 5,\nopponent, price"),
    ("Model", "P(plays) ×\npoints if he plays"),
    ("Optimizer", "£100m, formation,\nmax 3 per club"),
    ("Squad", "15 players,\n11 starters, captain"),
]


def main():
    width, height = 12, 2.3
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    card_width, card_height, header_height, gap = 2.0, 1.6, 0.5, 0.45
    left = (width - len(STEPS) * card_width - (len(STEPS) - 1) * gap) / 2
    bottom = 0.35
    last = len(STEPS) - 1

    for i, (title, description) in enumerate(STEPS):
        x = left + i * (card_width + gap)
        header_color = GREEN if i == last else SHADE
        header_bottom = bottom + card_height - header_height

        # white card with a coloured header (round at the top, square where it meets the body)
        ax.add_patch(FancyBboxPatch((x, bottom), card_width, card_height, boxstyle="round,pad=0,rounding_size=0.1",
                                    facecolor="white", edgecolor=LINE_GREY, linewidth=1.2, zorder=2))
        ax.add_patch(FancyBboxPatch((x, header_bottom), card_width, header_height,
                                    boxstyle="round,pad=0,rounding_size=0.1", facecolor=header_color,
                                    edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((x, header_bottom), card_width, 0.2, facecolor=header_color, edgecolor="none", zorder=3))

        ax.text(x + 0.18, header_bottom + header_height / 2, str(i + 1), color=GOLD, fontproperties=TITLE_FONT,
                fontsize=17, va="center", zorder=4)
        ax.text(x + card_width / 2 + 0.1, header_bottom + header_height / 2, title, color="white",
                fontproperties=TITLE_FONT, fontsize=17, ha="center", va="center", zorder=4)
        ax.text(x + card_width / 2, bottom + 0.55, description, color=TEXT_GREY, fontproperties=TEXT_FONT,
                fontsize=11.5, ha="center", va="center", linespacing=1.3, zorder=4)

        if i < last:
            ax.add_patch(FancyArrowPatch((x + card_width + 0.06, bottom + card_height / 2),
                                         (x + card_width + gap - 0.06, bottom + card_height / 2),
                                         arrowstyle="-|>", mutation_scale=15, color=ARROW_GREY, linewidth=1.8))

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=170, transparent=True)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
