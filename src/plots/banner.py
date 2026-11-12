"""
Draw the banner at the top of the README
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from src.config import ROOT
from src.plots.style import BOLD_FONT, GOLD, SHADE, TEXT_FONT, TITLE_FONT

OUTPUT_PATH = ROOT / "docs" / "banner.png"
PITCH_LIGHT = "#23853f"
PITCH_DARK = "#1f7a3a"
FORMATION = [
    (0.75, [1.7]),                    # goalkeeper
    (2.0, [0.65, 1.35, 2.05, 2.75]),  # defenders
    (3.6, [0.95, 1.7, 2.45]),         # midfielders
    (5.1, [0.85, 1.7, 2.55]),         # forwards
]
CAPTAIN = (5.1, 1.7)


def main():
    width, height = 12.8, 3.4
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    # grass in stripes
    stripes = 16
    for i in range(stripes):
        color = PITCH_DARK if i % 2 else PITCH_LIGHT
        ax.add_patch(Rectangle((i * width / stripes, 0), width / stripes, height, color=color, zorder=0))

    # pitch lines: outline, halfway line, centre circle, the right box
    line = dict(fill=False, edgecolor="white", linewidth=1.6, alpha=0.55, zorder=1)
    middle = height / 2
    ax.add_patch(Rectangle((0.25, 0.2), width - 0.5, height - 0.4, **line))
    ax.plot([width / 2, width / 2], [0.2, height - 0.2], color="white", linewidth=1.6, alpha=0.55, zorder=1)
    ax.add_patch(Circle((width / 2, middle), 0.62, **line))
    goal_line = width - 0.25  # only the right box, so no lines run behind the title
    ax.add_patch(Rectangle((goal_line, middle - 1.05), -1.35, 2.1, **line))
    ax.add_patch(Rectangle((goal_line, middle - 0.5), -0.5, 1.0, **line))

    # dark fade on the left, so the title is easy to read
    fade = np.ones((1, 256, 4))
    fade[..., :3] = np.array([11, 46, 23]) / 255   # the SHADE colour as numbers from 0 to 1
    fade[..., 3] = np.clip(np.linspace(0.75, -0.35, 256), 0, 1)
    ax.imshow(fade, extent=[0, width * 0.75, 0, height], aspect="auto", zorder=1)

    ax.text(0.75, 2.35, "FPL Points Predictor", color="white", fontproperties=TITLE_FONT, fontsize=44,
            va="center", zorder=3)
    ax.text(0.8, 1.47, "Predicts every player's points and picks\nthe best squad for the next gameweek",
            color="#d8f0dc", fontproperties=TEXT_FONT, fontsize=15, va="center", linespacing=1.3, zorder=3)
    ax.text(0.8, 0.83, "SQL  ·  machine learning  ·  mixed-integer programming", color="#9be9a8",
            fontproperties=BOLD_FONT, fontsize=12.5, va="center", zorder=3)

    # the squad: goalkeeper on the right, attacking to the left
    for x, ys in FORMATION:
        for y in ys:
            captain = (x, y) == CAPTAIN
            ax.add_patch(Circle((width - x, y), 0.17, facecolor=GOLD if captain else "white",
                                edgecolor=SHADE, linewidth=1.2, zorder=4))
            if captain:
                ax.text(width - x, y, "C", color=SHADE, fontproperties=BOLD_FONT, fontsize=10,
                        ha="center", va="center", zorder=5)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
