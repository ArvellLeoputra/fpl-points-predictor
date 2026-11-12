"""
Colours and fonts shared by the pictures of the README.
"""

from matplotlib.font_manager import FontProperties

from src.config import ROOT

FONTS = ROOT / "assets" / "fonts"   # Barlow, free under the SIL Open Font License
TITLE_FONT = FontProperties(fname=FONTS / "BarlowCondensed-Bold.ttf")
TEXT_FONT = FontProperties(fname=FONTS / "Barlow-Regular.ttf")
BOLD_FONT = FontProperties(fname=FONTS / "Barlow-SemiBold.ttf")

SHADE = "#0b2e17"
GREEN = "#1a9e4b"
GOLD = "#f2cc60"
TEXT_GREY = "#57606a"
LINE_GREY = "#d0d7de"
