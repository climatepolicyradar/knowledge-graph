# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""
Climate Policy Radar house style for matplotlib figures.

Applies the brand palette, typography and logo from the CPR presentation
guidelines. Import and call `apply()` once before drawing.

Typography: the brand fonts ship as *variable* TTFs, and matplotlib only ever
loads a variable font's default instance — so every weight would render at 400
and bold would silently fall back to a different family. `apply()` therefore
instantiates static Regular / SemiBold / Bold cuts with fontTools on first run
and caches them under assets/fonts/.

Colour: the classifier trio is Forest / Inky Blue / Mustard, assigned so the two
subconcepts take blue and yellow and their parent concept takes the green those
two mix to. Blue against Forest is the weak pair at worst-case CVD dE 21.5,
carried by direct labels and by band ordering in stacked charts.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import PathPatch
from svgpath2mpl import parse_path

BASE = Path(__file__).parent
ASSETS = BASE / "assets"
FONT_CACHE = ASSETS / "fonts"
LOGO = ASSETS / "cpr-logo.svg"

# ---- brand palette -------------------------------------------------------
WHITE = "#FFFFFF"
PAPER = "#FBFBF9"
RECYCLED_PAPER = "#E5E3DC"
LIGHT_BLUE = "#C6DDE8"
INKY_BLUE = "#1A4F8C"
INKY_NAVY = "#133051"
INKY_BLACK = "#101A24"
CARDBOARD = "#B7AE8E"
MUSTARD = "#C8B500"
FOREST = "#164F4A"
GREEN = "#A0AF54"

# Backgrounds are white throughout, per the brief.
SURFACE = WHITE
INK = INKY_BLACK
INK2 = "#4A5560"  # muted navy-grey for secondary text
INK3 = "#8A939C"  # tertiary text
GRID = "#E5E3DC"  # Recycled Paper doubles as the gridline tone

# Categorical trio for the three classifiers. The mapping is semantic: blue and
# yellow mix to green, so climate justice — the concept the other two sit under —
# takes Forest, and its two subconcepts take Inky Blue and Mustard.
#
# Blue against Forest is the one weak pair: worst-case CVD dE 21.5 (tritan) and
# near-identical lightness, L* 33 vs 30. Every mark carries a direct label, and
# stacked charts order the bands so those two are never adjacent.
Q32_C, Q911_C, Q912_C = FOREST, INKY_BLUE, MUSTARD
NEUTRAL = CARDBOARD  # "more than one label" / non-specific series

# Mustard and Green are legible as *fills* but fail as text on white (2.1:1 and
# 2.6:1). Text set in a series colour uses these darkened cousins instead; fills
# stay the brand values.
TEXT_ON_WHITE = {
    MUSTARD: "#8A7C00",
    GREEN: "#5F6E28",
    CARDBOARD: "#7A7259",
    "#8CB4CF": "#356F97",
}


def text_colour(fill: str) -> str:
    """Legible text tone for a series whose fill is `fill`."""
    return TEXT_ON_WHITE.get(fill, fill)


# Two-colour set where colour means country rather than classifier. Two shades
# of the brand blue rather than two hues: a second hue reads as a second
# meaning, and these figures already spend hue on the classifiers.
COUNTRY_A, COUNTRY_B = INKY_NAVY, "#8CB4CF"

# Sequential ramp, Light Blue -> Inky Blue -> Inky Navy.
BLUE_RAMP = [
    "#DCEBF2",
    "#C6DDE8",
    "#A9C9DC",
    "#8CB4CF",
    "#6F9FC2",
    "#5289B5",
    "#3E76A5",
    "#2F6497",
    "#1A4F8C",
    "#17427A",
    "#133051",
    "#0F2740",
]


def _instantiate_fonts() -> bool:
    """Cut static weights out of the variable brand fonts. Returns success."""
    families = {
        "Inter": "Inter-VariableFont_opsz,wght.ttf",
        "InterTight": "InterTight-VariableFont_wght.ttf",
        "GeistMono": "GeistMono-VariableFont_wght.ttf",
        "Newsreader": "Newsreader-VariableFont_opsz,wght.ttf",
    }
    search = [Path.home() / "Library/Fonts", Path("/Library/Fonts")]
    FONT_CACHE.mkdir(parents=True, exist_ok=True)
    made = False
    for family, filename in families.items():
        src = next((d / filename for d in search if (d / filename).exists()), None)
        if src is None:
            continue
        for weight in (400, 600, 700):
            out = FONT_CACHE / f"{family}-{weight}.ttf"
            if not out.exists():
                try:
                    from fontTools import ttLib
                    from fontTools.varLib import instancer

                    font = ttLib.TTFont(src)
                    axes = {"wght": weight}
                    if "opsz" in {a.axisTag for a in font["fvar"].axes}:
                        axes["opsz"] = 14
                    instancer.instantiateVariableFont(font, axes, inplace=True)
                    font.save(out)
                except Exception:
                    continue
            if out.exists():
                fm.fontManager.addfont(str(out))
                made = True
    return made


def apply() -> None:
    """
    Register fonts and set rcParams. Safe to call more than once.

    Sizing targets projection: figures are exported at roughly slide width, so
    body text sits at 11pt and nothing on a chart drops below about 8pt.
    """
    ok = _instantiate_fonts()
    stack = (
        ["Inter", "Inter Tight", "Helvetica Neue", "Arial", "DejaVu Sans"]
        if ok
        else ["Helvetica Neue", "Arial", "DejaVu Sans"]
    )
    if not ok:
        print("  ! brand fonts not found - falling back to system sans")
    mpl.rcParams.update(
        {
            "font.family": stack,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.size": 11,
            "svg.fonttype": "none",
        }
    )


TITLE_FONT = {"family": ["Inter Tight", "Inter", "Helvetica Neue", "Arial"]}


def sequential(value: float, vmax: float) -> str:
    if vmax <= 0:
        return BLUE_RAMP[0]
    i = int(round((value / vmax) ** 0.75 * (len(BLUE_RAMP) - 1)))
    return BLUE_RAMP[max(0, min(len(BLUE_RAMP) - 1, i))]


# Diverging ramp for ratios: red (depleted) through white (parity) to blue
# (enriched). Symmetric, six steps either side of white. Red sits outside the
# brand palette deliberately — mustard reads as the procedural-justice series
# and would collide with the column headers.
DIVERGING_RAMP = [
    "#B02418",
    "#C24A3D",
    "#D26F64",
    "#E0948C",
    "#EBB9B4",
    "#F6DFDC",
    "#FFFFFF",
    "#E4EFF6",
    "#C0DAEA",
    "#93BEDC",
    "#629FC9",
    "#3577AE",
    "#1A4F8C",
]


def diverging_ratio(ratio: float, max_log2: float) -> str:
    """
    Colour for a ratio on a log2 scale, so 0.5x is as far from 1 as 2x.

    A linear ramp would put 0.5 and 2.0 at wildly different distances from
    parity, which is wrong for a lift table: both are a factor of two.
    """
    import math

    if ratio <= 0 or max_log2 <= 0:
        return DIVERGING_RAMP[len(DIVERGING_RAMP) // 2]
    t = max(-1.0, min(1.0, math.log2(ratio) / max_log2))
    i = int(round((t + 1) / 2 * (len(DIVERGING_RAMP) - 1)))
    return DIVERGING_RAMP[i]


def ink_on(hexcolor: str) -> str:
    r, g, b = (int(hexcolor[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return WHITE if (0.299 * r + 0.587 * g + 0.114 * b) < 0.58 else INK


_LOGO_PATHS: list[tuple[str, str]] | None = None


def add_logo(fig, x: float = 0.012, y: float = 0.012, width: float = 0.115) -> None:
    """Draw the CPR wordmark into an inset axes at figure coordinates."""
    global _LOGO_PATHS
    if _LOGO_PATHS is None:
        if not LOGO.exists():
            _LOGO_PATHS = []
        else:
            _LOGO_PATHS = re.findall(
                r'<path d="([^"]+)"[^>]*fill="(#[0-9A-Fa-f]{6})"', LOGO.read_text()
            )
    if not _LOGO_PATHS:
        return
    height = width * (1021 / 17878) * (fig.get_figwidth() / fig.get_figheight())
    ax = fig.add_axes((x, y, width, height), zorder=10)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    for d, colour in _LOGO_PATHS:
        ax.add_patch(PathPatch(parse_path(d), facecolor=colour, linewidth=0))
    ax.set_xlim(0, 17878)
    ax.set_ylim(1021, 0)
    ax.set_aspect("equal")


SOURCE = "Source: climatepolicyradar.org, August 2026"


def titled(fig, title: str, how_to_read: str, note: str = "") -> None:
    """
    Title, a one-or-two line "how to read this" line, and a terse footnote.

    The analytical argument belongs in FINDINGS.md, not on the figure. Keep
    `how_to_read` to what the reader needs to decode the marks, and `note` to
    technical caveats that change how the numbers should be taken.
    """
    fig.text(
        0.012, 0.978, title, fontsize=21, weight=700, color=INK, va="top", **TITLE_FONT
    )
    fig.text(
        0.012, 0.930, how_to_read, fontsize=12.5, color=INK2, va="top", linespacing=1.28
    )
    footer = f"{note}\n{SOURCE}" if note else SOURCE
    fig.text(0.145, 0.014, footer, fontsize=9, color=INK3, va="bottom", linespacing=1.5)


def save(fig, name: str, figdir: Path) -> None:
    add_logo(fig)
    for ext in ("png", "svg"):
        fig.savefig(
            figdir / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE
        )
    plt.close(fig)
    print(f"  wrote figures/{name}.png / .svg")
