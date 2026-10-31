"""
FIGURE 1 - METHODOLOGY WORKFLOW
Keyword-led: labels and numbers only, no sentences. Five stages read top to
bottom; the held-out test is routed down the right margin to show it is opened
once, at the end. 720 dpi, no on-figure title.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.environ.get("FIGDIR", "/tmp/wf")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"savefig.dpi": 720, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.05, "figure.facecolor": "white"})

# Times New Roman, with metric-compatible fallbacks so the figures render the
# same on machines that do not ship it. Nothing exceeds 12 pt.
TIMES_STACK = ["Times New Roman", "Liberation Serif", "Nimbus Roman",
               "DejaVu Serif", "serif"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": TIMES_STACK,
    "mathtext.fontset": "stix", "mathtext.rm": "Times New Roman",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

SAT, MET, URB, GND = "#2F6FB0", "#4E9A68", "#D18A33", "#A8473E"
MODEL, TEST, PROD, RULE = "#44596B", "#7C4F9E", "#2E6E52", "#6E6A86"
GREY, FAINT = "#5E6A72", "#9AA4AC"
R = dict(boxstyle="round,pad=0.34,rounding_size=0.12", linewidth=0)

fig, ax = plt.subplots(figsize=(13.2, 8.0))
ax.set_xlim(0, 106); ax.set_ylim(0, 100); ax.axis("off")


def card(x, y, w, h, title, lines, fc, tfs=9.2, lfs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=fc, **R))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center",
            fontsize=tfs, color="white", fontweight="bold", zorder=5)
    ax.text(x + w / 2, y + (h - 5.2) / 2 + 0.4, "\n".join(lines), ha="center",
            va="center", fontsize=lfs, color="#E6ECF0", linespacing=1.55, zorder=5)


def arrow(x1, y1, x2, y2, c=GREY, lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=lw, color=c,
                                 shrinkA=2, shrinkB=2, zorder=2))


def band(y, n, label):
    ax.text(1.0, y, n, ha="left", va="center", fontsize=12,
            fontweight="bold", color="#CBD3D9")
    ax.text(4.2, y, label, ha="left", va="center", fontsize=9.6,
            fontweight="bold", color="#2A3A45", linespacing=1.3)


# -------------------------------------------------------------- 1 inputs --
band(90.5, "1", "INPUTS")
for i, (t, lines, c) in enumerate([
        ("TROPOMI", ["NO$_2$ column", "5.5 \u00d7 3.5 km"], SAT),
        ("MERRA-2", ["PBLH \u00b7 T \u00b7 wind", "\u2248 55 km"], MET),
        ("GHSL", ["built-up \u00b7 population", "100 m"], URB),
        ("GROUND", ["EPA \u00b7 CNEMC", "CPCB \u00b7 ADQCC"], GND)]):
    card(21 + i * 21, 85.0, 19.2, 11.0, t, lines, c)

# -------------------------------------------------------- 2 harmonise --
band(74.5, "2", "HARMONISE")
ax.add_patch(FancyBboxPatch((21, 69.5), 61.2, 9.6, facecolor=MODEL,
                            edgecolor=MODEL, **R))
ax.text(51.6, 76.5, "20 km airshed   \u00b7   monthly   \u00b7   ppb", ha="center",
        va="center", fontsize=10.4, color="white", fontweight="bold", zorder=5)
ax.text(51.6, 72.4, "one unit contract, asserted at every stage", ha="center",
        va="center", fontsize=7.4, color="#C3D0D8", style="italic", zorder=5)
card(84.2, 69.5, 19.2, 9.6, "OVERPASS", ["10:00\u201314:00 local"], GND, tfs=9)
for x in (30.6, 51.6, 72.6):
    arrow(x, 85.0, x, 79.1)
arrow(93.8, 85.0, 93.8, 79.1)

# --------------------------------------------------------- 3 datasets --
band(58.5, "3", "DATASETS")
card(21, 52.5, 19.2, 11.6, "A \u00b7 TARGET", ["100 cities", "no ground truth"], "#6E8296")
card(42.2, 52.5, 40.0, 11.6, "B \u00b7 TRAINING",
     ["49 cities   \u00b7   15 US  \u00b7  20 China  \u00b7  10 India  \u00b7  4 Gulf"], MODEL)
card(84.2, 52.5, 19.2, 11.6, "C \u00b7 HELD OUT", ["5 arid cities", "opened once"], TEST)
arrow(46, 69.5, 30.6, 64.1)
arrow(57, 69.5, 62.2, 64.1)
arrow(93.8, 69.5, 93.8, 64.1)
ax.text(51.6, 49.8, "128 distinct cities   \u00b7   54 with ground observations   \u00b7   "
        "zero train\u2013test overlap", ha="center", fontsize=7.4, color=FAINT,
        style="italic")

# --------------------------------------------------------- 4 modelling --
band(38.5, "4", "MODEL")
card(21, 29.0, 19.2, 14.0, "PROTOCOL",
     ["fixed before modelling", "", "P1   city-blocked", "P2   region-blocked",
      "P3   temporal"], RULE, tfs=9, lfs=7.3)
card(42.2, 37.0, 40.0, 7.6, "STAGE 1",
     ["35 strong-link cities   \u00b7   $\\rho$ = 0.83 \u2013 0.92"], MODEL, tfs=9.4)
card(42.2, 27.2, 40.0, 7.6, "STAGE 2",
     ["14 hot cities   \u00b7   residual   \u00b7   gated 23\u201325 \u00b0C"], "#8A6A42", tfs=9.4)
card(84.2, 30.0, 19.2, 13.0, "DIAGNOSE",
     ["column ablation", "SHAP", "10-seed ensemble"], "#5A6B79", tfs=9, lfs=7.4)
arrow(62.2, 52.5, 62.2, 44.6)
arrow(62.2, 37.0, 62.2, 34.8)
arrow(40.2, 36.0, 42.2, 36.0)
arrow(82.2, 36.5, 84.2, 36.5)

# ------------------------------------------------------------ 5 result --
band(16.0, "5", "RESULT")
card(21, 9.0, 25.4, 13.0, "SKILL",
     ["$R^2$ = 0.57   temperate", "$R^2$ = 0.15   arid",
      "bias  \u22127.44 \u2192 \u22120.23 ppb"], PROD, tfs=9.4, lfs=7.8)
card(48.6, 9.0, 25.4, 13.0, "UNCERTAINTY",
     ["$\\rho$ = tanh($aT + b$)", "$r$ = \u22120.707,  $n$ = 54",
      "skill from climate"], "#35795C", tfs=9.4, lfs=7.8)
card(76.2, 9.0, 27.2, 13.0, "PRODUCT",
     ["surface NO$_2$  \u00b7  100 cities", "two confidence axes",
      "domain  \u00b7  expected $\\rho$"], PROD, tfs=9.4, lfs=7.8)
arrow(62.2, 27.2, 62.2, 22.0)
arrow(93.8, 30.0, 90, 22.0)
arrow(46.4, 15.5, 48.6, 15.5)
arrow(74, 15.5, 76.2, 15.5)

# held-out path, down the right margin
ax.plot([103.4, 104.8], [58.3, 58.3], color=TEST, lw=1.3, zorder=2)
ax.plot([104.8, 104.8], [58.3, 16.6], color=TEST, lw=1.3, zorder=2)
arrow(104.8, 16.6, 103.4, 15.5, c=TEST, lw=1.3)
ax.text(103.8, 37.0, "held out until the end", fontsize=7.2, color=TEST,
        ha="center", va="center", rotation=90, style="italic")

fig.savefig(os.path.join(OUT, "Fig00_methodology_workflow.png"))
plt.close(fig)
print("wrote Fig00_methodology_workflow.png")
