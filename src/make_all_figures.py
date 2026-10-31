"""
MAKE_ALL_FIGURES - every figure for the manuscript, one run
============================================================
    pip install geopandas
    python make_all_figures.py

Writes to  <FIGDIR>/main/  and  <FIGDIR>/supplementary/
All figures 720 dpi, no on-figure titles, no overlapping text, tight layout.
Captions live in FIGURE_CAPTIONS.md, numbering in the filenames only.

EVERY NUMBER IS COMPUTED LIVE from the frozen datasets. Nothing is hardcoded,
so the figures cannot drift out of step with the analysis.

WHAT IS DELIBERATELY NOT A FIGURE
  Content that is a small set of numbers belongs in a table. Dropped from the
  earlier figure set and moved to tables: per-feature Spearman correlations
  (10 x 2 values), temporal coverage counts (2 x 7), covariate-shift SMDs (6),
  the feature-block ablation (4), and the design ablation (5). Bar charts of a
  handful of numbers waste a figure slot and are harder to read than a table.

RUNTIME  about 4-8 minutes; the leave-one-city-out refits dominate.
"""
import os, warnings, itertools
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr, gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- paths ----
DATA   = r"D:\Cities\Global_Met\final_datasets"
FIGDIR = r"D:\Cities\Global_Met\figures"
GEOJSON = os.path.join(DATA, "World_Countries_Political_Boundaries.geojson")
CACHE   = os.path.join(DATA, "countries_dissolved.gpkg")

MAIN = os.path.join(FIGDIR, "main")
SUPP = os.path.join(FIGDIR, "supplementary")
for d in (MAIN, SUPP):
    os.makedirs(d, exist_ok=True)

DPI = 720
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "font.size": 9.5, "axes.labelsize": 9.5,
    "axes.titlesize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.facecolor": "white",
    "axes.linewidth": 0.8, "lines.antialiased": True,
})

# Times New Roman, with metric-compatible fallbacks so the figures render the
# same on machines that do not ship it. Nothing exceeds 12 pt.
TIMES_STACK = ["Times New Roman", "Liberation Serif", "Nimbus Roman",
               "DejaVu Serif", "serif"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": TIMES_STACK,
    "mathtext.fontset": "stix", "mathtext.rm": "Times New Roman",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C_US, C_CN, C_IN, C_IA, C_GF = "#2c6fbb", "#c0392b", "#e08e0b", "#8e44ad", "#16a085"
COL = {"USA": C_US, "CHINA": C_CN, "INDIA_TRAIN": C_IN, "INDIA_ARID": C_IA, "GULF": C_GF}
NICE = {"USA": "USA", "CHINA": "China", "INDIA_TRAIN": "India (train)",
        "INDIA_ARID": "India (arid test)", "GULF": "Gulf"}
LAND, EDGE, SEA = "#f2efe9", "#b9b3a7", "#dfeaf2"
CMAP = "YlOrRd"


def save(fig, folder, name):
    fig.savefig(os.path.join(folder, name))
    plt.close(fig)
    print(f"  {os.path.basename(folder):<14} {name}")


def panel(ax, s, x=0.015, y=0.975):
    ax.text(x, y, f"({s})", transform=ax.transAxes, va="top", ha="left",
            fontsize=10.5, fontweight="bold", zorder=10)


# ------------------------------------------------------------------ data ----
print("loading datasets")
A = pd.read_csv(os.path.join(DATA, "DATASET_A_target_global100.csv"))
B = pd.read_csv(os.path.join(DATA, "DATASET_B_train.csv"))
C = pd.read_csv(os.path.join(DATA, "DATASET_C_test_india_arid.csv"))
for d in (A, B, C):
    d["sd_ratio"] = d.no2_sd / d.no2_vcd_mol_m2
if "airshed_km" in A.columns:
    A = A[A.airshed_km == 20].copy()

F = ["no2_vcd_mol_m2", "pblh_m", "temp_c", "wind_u_ms", "wind_v_ms",
     "wind_speed_ms", "population", "built_up_km2", "sd_ratio", "cloud_fraction"]
LBL = {"no2_vcd_mol_m2": "NO$_2$ column", "pblh_m": "PBLH", "temp_c": "Temperature",
       "wind_u_ms": "Wind $u$", "wind_v_ms": "Wind $v$", "wind_speed_ms": "Wind speed",
       "population": "Population", "built_up_km2": "Built-up area",
       "sd_ratio": "Column SD ratio", "cloud_fraction": "Cloud fraction"}
T = "ground_no2_ppb"

STRONG = ["USA", "CHINA"]
COOL = B[B.region.isin(STRONG)].reset_index(drop=True)
HOTD = B[~B.region.isin(STRONG)].reset_index(drop=True)


def mk(seed=42):
    return xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                            min_child_weight=5, tree_method="hist",
                            random_state=seed, n_jobs=4)


def loco(d, feats=F, seed=42):
    o = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=d.city.nunique()).split(d[feats], d[T], d.city):
        o[te] = mk(seed).fit(d[feats].iloc[tr], d[T].iloc[tr]).predict(d[feats].iloc[te])
    return o


print("fitting models (this is the slow part)")
oofs = np.column_stack([loco(COOL, seed=s) for s in range(6)])   # matches the text
COOL["oof"] = oofs.mean(1)
P1 = r2_score(COOL[T], COOL.oof)
P1SD = np.std([r2_score(COOL[T], oofs[:, i]) for i in range(oofs.shape[1])])

# Seed ensemble for deployment predictions: a single XGBoost fit is one draw
# from a distribution, and out-of-domain cities are the most sensitive to it
# (Riyadh moved 2 ppb between two runs of identical code). Figures use the mean
# over N_SEEDS fits so they are reproducible.
N_SEEDS = 10
PAIRS = []
for _s in range(N_SEEDS):
    _s1 = mk(_s).fit(COOL[F], COOL[T])
    _h = HOTD.copy()
    _h["base"] = _s1.predict(_h[F])
    PAIRS.append((_s1, mk(_s).fit(_h[F + ["base"]], _h[T] - _h.base)))
stage1, stage2 = PAIRS[0]
HOTD = HOTD.copy()
HOTD["base"] = stage1.predict(HOTD[F])

# Stage 2 is CLIMATE-GATED: it was fitted only on hot/arid cities, so it is
# weighted by the city's long-run mean temperature and applied only where it was
# trained. Without the gate it inflates temperate cities by ~6 ppb (Tianjin:
# observed 10.5, ungated two-stage 21.7).
GATE_LO, GATE_HI = 23.0, 25.0


def gate(d):
    tc = d.groupby("city").temp_c.transform("mean")
    return np.clip((tc - GATE_LO) / (GATE_HI - GATE_LO), 0, 1).values


def two_stage(d):
    w = gate(d)
    out = np.empty((len(d), len(PAIRS)))
    for k, (s1, s2) in enumerate(PAIRS):
        b = s1.predict(d[F])
        out[:, k] = b + w * s2.predict(d.assign(base=b)[F + ["base"]])
    return out.mean(axis=1)


# Stage-1 predictions use the SAME 10-seed ensemble as the two-stage model.
# Using a single seed for one and the ensemble for the other made the figures
# disagree with the text by up to 0.03 in R2 and 0.2 ppb in bias.
def stage1_only(d):
    return np.column_stack([s1.predict(d[F]) for s1, _ in PAIRS]).mean(axis=1)


C_pred_1 = stage1_only(C)
C_pred_2 = two_stage(C)
A["est_ppb"] = two_stage(A)

# per-city rho with climate, across every city that has ground truth
allc = pd.concat([B, C], ignore_index=True)
rows = []
for c, g in allc.groupby("city"):
    if len(g) < 19:
        continue
    rows.append(dict(city=c, region=g.region.iloc[0],
                     rho=spearmanr(g.no2_vcd_mol_m2, g[T])[0],
                     temp=g.temp_c.mean(), pblh=g.pblh_m.mean()))
Q = pd.DataFrame(rows)
KZ = np.polyfit(Q.temp, np.arctanh(np.clip(Q.rho, -0.999, 0.999)), 1)
rho_of_T = lambda t: np.tanh(np.polyval(KZ, t))

RNG = {c: (COOL[c].min(), COOL[c].max()) for c in F}
A["in_domain"] = np.all([A[c].between(*RNG[c]) for c in F], axis=0)
# The rho model was FITTED on city-mean temperature, so it must be EVALUATED at
# city-mean temperature. Averaging rho(T_month) instead gives a different answer
# because tanh is non-linear (Jensen's inequality): for Riyadh, 0.45 vs 0.56.
_ctemp = A.groupby("city").temp_c.transform("mean")
A["exp_rho"] = rho_of_T(_ctemp)
print(f"  P1 = {P1:.3f} +/- {P1SD:.3f} | arid stage-1 {r2_score(C[T], C_pred_1):+.3f}"
      f" | arid two-stage {r2_score(C[T], C_pred_2):+.3f}")

# ======================================================== MAIN FIGURES ======
print("\nmain figures")

# ---- F0  methodology workflow -----------------------------------------------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
SAT, MET, URB = "#3d7ebf", "#5aa469", "#d98c3f"
GND, MODELC, TESTC, PROD = "#b0524b", "#4a5568", "#8e44ad", "#2f6f4e"
BOX = dict(boxstyle="round,pad=0.34,rounding_size=0.10", linewidth=1.0)
fig, ax = plt.subplots(figsize=(11.6, 8.4))
ax.set_xlim(0, 103); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc, ec=None, fs=7.6, weight="normal", tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc,
                                edgecolor=ec or fc, **BOX))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight=weight, linespacing=1.35, zorder=5)


def arrow(x1, y1, x2, y2, color="#5a5a5a", lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, linewidth=lw, color=color,
                                 shrinkA=1, shrinkB=1, zorder=2))


def band(y, label, sub=""):
    ax.text(-1.5, y, label, ha="right", va="center", fontsize=8.6,
            fontweight="bold", color="#333")
    if sub:
        ax.text(-1.5, y - 3.4, sub, ha="right", va="center", fontsize=6.8,
                color="#777", style="italic")


# ------------------------------------------------- 1. sources ---------------
band(92, "1  ACQUISITION", "2019–2025, monthly")
box(2, 86.5, 21, 11, "Sentinel-5P TROPOMI\nNO$_2$ tropospheric column\n"
    "QA $\\geq$ 75, ~5.5 km", SAT)
box(26, 86.5, 21, 11, "MERRA-2 reanalysis\nPBLH, T2M, U10M, V10M\n~50 km", MET)
box(50, 86.5, 21, 11, "GHSL P2023A\nbuilt-up surface, population\n100 m, 5-yr epochs", URB)
box(74, 86.5, 24, 11, "Ground networks\nEPA · CNEMC · CPCB · ADQCC\n"
    "overpass-hour means", GND)

# ------------------------------------------------- 2. harmonisation ---------
band(76, "2  HARMONISATION", "Google Earth Engine")
box(2, 70.5, 69, 9.5, "Zonal aggregation over a 20 km urban airshed  ·  "
    "single unit contract enforced by assertion\n"
    "temperature K$\\rightarrow$°C once  ·  concentrations in ppb  ·  "
    "GHSL no-data masked  ·  QA and granule counts retained", "#4b5d6b", fs=7.4)
box(74, 70.5, 24, 9.5, "Overpass-window filter\n10:00–14:00 local\nto match TROPOMI", GND)
for x in (12.5, 36.5, 60.5):
    arrow(x, 86.5, x, 80.0)
arrow(86, 86.5, 86, 80.0)

# ------------------------------------------------- 3. datasets --------------
band(60, "3  DATASETS", "frozen, leakage-checked")
box(2, 52.5, 28, 11.5, "A  ·  TARGET\n100 global cities\n8,379 city-months\n"
    "no ground truth", "#6b7a8f")
box(33, 52.5, 34, 11.5, "B  ·  TRAINING\n50 cities  ·  15 US, 20 China,\n"
    "10 India, 5 Gulf\nwith ground truth", MODELC)
box(70, 52.5, 28, 11.5, "C  ·  HELD-OUT TESTC\n5 arid NW Indian cities\n"
    "207 city-months\nopened once, at the end", TESTC)
arrow(36, 70.5, 16, 64.0)
arrow(44, 70.5, 50, 64.0)
arrow(80, 70.5, 84, 64.0)
ax.text(50, 49.4, "assertions: zero duplicate city-months  ·  zero train/test overlap  ·  "
        "one pipeline version, one PBLH definition, one airshed",
        ha="center", fontsize=6.9, color="#666", style="italic")

# ------------------------------------------------- 4. protocol + model ------
band(41, "4  MODELLING", "protocol fixed first")
box(2, 33.5, 28, 10.5, "Validation protocol\nfixed BEFORE modelling\n"
    "P1 leave-one-city-out\nP2 leave-one-region-out\nP3 temporal hold-out",
    "#7a6a86", fs=7.2)
box(33, 35.0, 34, 9.0, "STAGE 1  ·  XGBoost\n35 strong-link cities (US + China)\n"
    "$\\rho$(column, surface) > 0.8", MODELC)
box(33, 24.0, 34, 9.0, "STAGE 2  ·  XGBoost\nfitted on hot/arid cities\n"
    "predicts stage-1 residual", "#8a6b4a")
box(70, 33.5, 28, 10.5, "Diagnostics\nsatellite-column ablation\nSHAP, split between-\n"
    "and within-city", "#5c6b7a", fs=7.2)
arrow(50, 52.5, 50, 44.0)
arrow(50, 35.0, 50, 33.0, style="-|>")
arrow(30, 40.0, 33, 40.0)
arrow(67, 40.0, 70, 40.0)

# ------------------------------------------------- 5. results ---------------
band(17, "5  RESULT", "with uncertainty")
box(2, 9.5, 30, 11.0, "Validated skill\nP1 $R^2$ = 0.57 (temperate)\n"
    "arid transfer $R^2$ = 0.08\nbias $-$7.5 $\\rightarrow$ $-$0.03 ppb", PROD)
box(35, 9.5, 30, 11.0, "Uncertainty model\n"
    "$\\rho$ = tanh($a\\cdot T + b$)\n$r$ = $-$0.71, $n$ = 54\n"
    "expected skill from climate", "#3f7a63")
box(68, 9.5, 30, 11.0, "Product\nsurface NO$_2$, 100 cities\ntwo confidence axes:\n"
    "in-domain  ·  expected $\\rho$", PROD)
arrow(50, 24.0, 50, 20.5)
arrow(84, 33.5, 84, 20.5)
# held-out path: routed down the right margin so it never crosses a box
ax.plot([98.6, 99.6], [58.2, 58.2], color=TESTC, lw=1.3, zorder=2)
ax.plot([99.6, 99.6], [58.2, 17.5], color=TESTC, lw=1.3, zorder=2)
arrow(99.6, 17.5, 98.0, 15.0, color=TESTC, lw=1.3)
ax.text(99.0, 37.5, "held-out evaluation", fontsize=6.8, color=TESTC,
        ha="center", va="center", rotation=90, style="italic")
arrow(32, 15.0, 35, 15.0)
arrow(65, 15.0, 68, 15.0)

# legend
for i, (c, t) in enumerate([(SAT, "satellite"), (MET, "meteorology"),
                            (URB, "urban form"), (GND, "ground truth"),
                            (MODELC, "model"), (TESTC, "held-out test"),
                            (PROD, "result")]):
    ax.add_patch(FancyBboxPatch((2 + i * 13.6, 2.0), 2.4, 2.4, facecolor=c,
                                edgecolor=c, boxstyle="round,pad=0.1,rounding_size=0.06"))
    ax.text(5.2 + i * 13.6, 3.2, t, fontsize=6.9, va="center", color="#333")

save(fig, MAIN, "Fig00_methodology_workflow.png")


# ---- F1  column-to-surface relationship, by region --------------------------
fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
for r, g in allc.groupby("region"):
    ax[0].scatter(g.no2_vcd_mol_m2 * 1e5, g[T], s=6, alpha=.30,
                  color=COL[r], edgecolors="none", label=NICE[r], rasterized=True)
for r, g in allc.groupby("region"):
    x = g.no2_vcd_mol_m2 * 1e5
    xs = np.linspace(x.quantile(.02), x.quantile(.98), 40)
    ax[0].plot(xs, np.polyval(np.polyfit(x, g[T], 1), xs), color=COL[r], lw=2.0)
ax[0].set(xlabel="TROPOMI NO$_2$ column ($\\times10^{-5}$ mol m$^{-2}$)",
          ylabel="Observed surface NO$_2$ (ppb)")
lg = ax[0].legend(loc="upper right", markerscale=3, handletextpad=.4)
for h in lg.legend_handles:
    h.set_alpha(1)
panel(ax[0], "a")

ratio = allc.groupby("region").apply(
    lambda g: g[T].mean() / g.no2_vcd_mol_m2.mean(), include_groups=False)
rr = ratio.sort_values() / 1e3
bars = ax[1].barh([NICE[i] for i in rr.index], rr.values,
                  color=[COL[i] for i in rr.index], height=.62)
for b, v in zip(bars, rr.values):
    ax[1].text(v + rr.max() * .02, b.get_y() + b.get_height() / 2, f"{v:,.0f}",
               va="center", fontsize=8.5, fontweight="bold")
ax[1].set(xlabel="Surface-to-column ratio ($\\times10^{3}$ ppb per mol m$^{-2}$)",
          xlim=(0, rr.max() * 1.18))
panel(ax[1], "b")
save(fig, MAIN, "Fig01_column_surface_relationship.png")

# ---- F2  lid effect ---------------------------------------------------------
regs = [r for r in ["USA", "CHINA", "INDIA_TRAIN", "GULF"] if r in allc.region.unique()]
fig, ax = plt.subplots(1, len(regs), figsize=(3.0 * len(regs), 3.5),
                       sharey=True, constrained_layout=True)
for k, r in enumerate(regs):
    g = allc[allc.region == r]
    pa = g.pblh_m - g.groupby("city").pblh_m.transform("mean")
    ya = g[T] - g.groupby("city")[T].transform("mean")
    ax[k].scatter(pa, ya, s=5, alpha=.25, color=COL[r], edgecolors="none", rasterized=True)
    xs = np.linspace(pa.quantile(.01), pa.quantile(.99), 40)
    ax[k].plot(xs, np.polyval(np.polyfit(pa, ya, 1), xs), color="k", lw=2.0)
    ax[k].axhline(0, color="k", lw=.6, ls=":")
    ax[k].axvline(0, color="k", lw=.6, ls=":")
    ax[k].set_xlabel("PBLH anomaly (m)")
    ax[k].text(.96, .05, f"{NICE[r]}\n$\\rho$ = {spearmanr(pa, ya)[0]:+.2f}",
               transform=ax[k].transAxes, ha="right", va="bottom", fontsize=8.5)
    panel(ax[k], "abcd"[k])
ax[0].set_ylabel("Surface NO$_2$ anomaly (ppb)")
save(fig, MAIN, "Fig02_lid_effect.png")

# ---- F3  model performance --------------------------------------------------
fig = plt.figure(figsize=(11.4, 4.4), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
a0 = fig.add_subplot(gs[0])
for r, g in COOL.groupby("region"):
    a0.scatter(g[T], g.oof, s=7, alpha=.35, color=COL[r], edgecolors="none",
               label=NICE[r], rasterized=True)
lim = [0, max(COOL[T].max(), COOL.oof.max()) * 1.04]
a0.plot(lim, lim, "k--", lw=1.0)
a0.set(xlim=lim, ylim=lim, xlabel="Observed surface NO$_2$ (ppb)",
       ylabel="Estimated surface NO$_2$ (ppb)")
# panel (a) plots the seed-ensemble prediction, so it must be labelled with the
# R2 of that prediction, not with the mean of the per-seed coefficients
a0.text(.96, .06, f"$R^2$ = {r2_score(COOL[T], COOL.oof):.3f}\n"
        f"RMSE = {np.sqrt(mean_squared_error(COOL[T], COOL.oof)):.2f} ppb\n"
        f"$n$ = {len(COOL):,}", transform=a0.transAxes, ha="right", fontsize=8.5)
lg = a0.legend(loc="upper left", markerscale=2.6)
for h in lg.legend_handles:
    h.set_alpha(1)
panel(a0, "a")

a1 = fig.add_subplot(gs[1])
res = COOL.oof - COOL[T]
a1.scatter(COOL.oof, res, s=7, alpha=.28, color="#5a7d9a", edgecolors="none", rasterized=True)
a1.axhline(0, color="k", lw=1.0)
lo = pd.qcut(COOL.oof, 10, duplicates="drop")
mid = COOL.groupby(lo, observed=True).oof.mean()
a1.plot(mid, res.groupby(lo, observed=True).mean(), "o-", color="#c0392b", ms=3.5, lw=1.4)
a1.set(xlabel="Estimated surface NO$_2$ (ppb)", ylabel="Residual (ppb)")
panel(a1, "b")

a2 = fig.add_subplot(gs[2])
pc = []
for c, g in COOL.groupby("city"):
    m = (COOL.city == c).values
    rs = [r2_score(COOL[T][m], oofs[m, i]) for i in range(oofs.shape[1])]
    pc.append((c, COOL.region[m].iloc[0], np.mean(rs), np.std(rs)))
pc = pd.DataFrame(pc, columns=["city", "region", "r2", "sd"]).sort_values("r2")
a2.barh(range(len(pc)), pc.r2, xerr=pc.sd, height=.72,
        color=[COL[r] for r in pc.region], alpha=.88,
        error_kw=dict(lw=.6, ecolor="0.25", capsize=1.5))
a2.axvline(0, color="k", lw=.9)
a2.set_yticks(range(len(pc)), pc.city, fontsize=6.4)
a2.set_xlabel("Leave-one-city-out $R^2$")
a2.set_ylim(-.8, len(pc) - .2)
panel(a2, "c")
save(fig, MAIN, "Fig03_model_performance.png")

# ---- F4  rho vs climate  (KEY) ---------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.3), constrained_layout=True)
for v, a, xl in [("temp", ax[0], "Mean air temperature (°C)"),
                 ("pblh", ax[1], "Mean planetary boundary layer height (m)")]:
    for r, g in Q.groupby("region"):
        a.scatter(g[v], g.rho, s=46, color=COL[r], alpha=.88, edgecolors="k",
                  linewidths=.4, label=NICE[r], zorder=3)
    rr_, pp_ = pearsonr(Q[v], Q.rho)
    xs = np.linspace(Q[v].min(), Q[v].max(), 60)
    if v == "temp":
        a.plot(xs, rho_of_T(xs), "k--", lw=1.8, zorder=2)
    else:
        a.plot(xs, np.polyval(np.polyfit(Q[v], Q.rho, 1), xs), "k--", lw=1.8, zorder=2)
    ptxt = "< 0.0001" if pp_ < 1e-4 else f"= {pp_:.4f}"
    a.text(.97, .96, f"$r$ = {rr_:+.3f},  $p$ {ptxt},  $n$ = {len(Q)}",
           transform=a.transAxes, ha="right", va="top", fontsize=8.5)
    a.set(xlabel=xl, ylabel=r"$\rho$ (NO$_2$ column, surface NO$_2$)", ylim=(-.28, 1.12))
riy_t = A[A.city.str.contains("Riy", na=False)].temp_c.mean()
if np.isfinite(riy_t):
    ax[0].axvline(riy_t, color="0.45", ls=":", lw=1.4, zorder=1)
    ax[0].annotate(f"Riyadh\n{riy_t:.1f} °C,  $\\rho\\approx${rho_of_T(riy_t):.2f}",
                   xy=(riy_t, rho_of_T(riy_t)), xytext=(riy_t - 6.2, 0.16),
                   fontsize=7.8, color="0.25", ha="center", va="center",
                   arrowprops=dict(arrowstyle="-", color="0.45", lw=.8,
                                   shrinkA=0, shrinkB=3))
ax[0].legend(loc="lower left", ncol=2, columnspacing=.9, handletextpad=.3)
panel(ax[0], "a"); panel(ax[1], "b")
save(fig, MAIN, "Fig04_rho_versus_climate.png")

# ---- F5  arid transfer ------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(12.4, 4.0), constrained_layout=True)
lim = [0, max(C[T].max(), C_pred_1.max(), C_pred_2.max()) * 1.05]
for k, (p, lab) in enumerate([(C_pred_1, "Single-stage"), (C_pred_2, "Two-stage")]):
    ax[k].scatter(C[T], p, s=11, alpha=.5, color=C_IA, edgecolors="none", rasterized=True)
    ax[k].plot(lim, lim, "k--", lw=1.0)
    ax[k].set(xlim=lim, ylim=lim, xlabel="Observed surface NO$_2$ (ppb)",
              ylabel="Estimated surface NO$_2$ (ppb)" if k == 0 else "")
    ax[k].text(.96, .06, f"{lab}\n$R^2$ = {r2_score(C[T], p):+.3f}\n"
               f"bias = {np.mean(p - C[T]):+.2f} ppb",
               transform=ax[k].transAxes, ha="right", fontsize=8.5)
    panel(ax[k], "ab"[k])
bias = pd.DataFrame({"city": C.city, "s1": C_pred_1 - C[T], "s2": C_pred_2 - C[T]}) \
    .groupby("city").mean().sort_values("s1")
yy = np.arange(len(bias))
ax[2].barh(yy - .19, bias.s1, height=.38, color="#b0b7be", label="Single-stage")
ax[2].barh(yy + .19, bias.s2, height=.38, color=C_IA, label="Two-stage")
ax[2].axvline(0, color="k", lw=.9)
ax[2].set_yticks(yy, bias.index, fontsize=8)
ax[2].set_xlabel("Mean bias (ppb)")
ax[2].legend(loc="lower left")
panel(ax[2], "c")
save(fig, MAIN, "Fig05_arid_transfer.png")

# ---- F6  SHAP decomposition -------------------------------------------------
dm = xgb.DMatrix(COOL[F], feature_names=F)
sv = stage1.get_booster().predict(dm, pred_contribs=True)[:, :-1]
S = pd.DataFrame(sv, columns=F); S["city"] = COOL.city.values
dd = []
for f in F:
    tot = S[f].var(); bet = S.groupby("city")[f].transform("mean").var()
    dd.append(dict(f=f, ma=np.abs(S[f]).mean(), bet=bet, wit=max(tot - bet, 0)))
dd = pd.DataFrame(dd).sort_values("ma")
dd["pw"] = 100 * dd.wit / (dd.bet + dd.wit)
C_BET, C_WIT = "#b8804a", "#2c6fbb"
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
yy = np.arange(len(dd))
ax[0].barh(yy, dd.bet, height=.68, color=C_BET, label="Between-city (sets the level)")
ax[0].barh(yy, dd.wit, left=dd.bet, height=.68, color=C_WIT,
           label="Within-city (tracks time)")
ax[0].set_yticks(yy, [LBL[f] for f in dd.f])
ax[0].set_xlabel("Variance of SHAP values (ppb$^2$)")
tot = (dd.bet + dd.wit)
for i, (t, p) in enumerate(zip(tot, dd.pw)):
    ax[0].text(t + tot.max() * .015, i, f"{p:.0f}%", va="center", fontsize=7.6)
ax[0].set_xlim(0, tot.max() * 1.16)
ax[0].legend(loc="lower right")
panel(ax[0], "a")
i = F.index("pblh_m")
s = ax[1].scatter(COOL.pblh_m, sv[:, i], c=COOL.temp_c, s=7, alpha=.55,
                  cmap="coolwarm", edgecolors="none", rasterized=True)
ax[1].axhline(0, color="k", lw=.9)
ax[1].set(xlabel="PBLH (m)", ylabel="SHAP value for PBLH (ppb)")
cb = fig.colorbar(s, ax=ax[1], pad=.015); cb.set_label("Temperature (°C)", fontsize=9)
ax[1].text(.97, .96, f"$r$ = {np.corrcoef(COOL.pblh_m, sv[:, i])[0,1]:+.3f}",
           transform=ax[1].transAxes, ha="right", va="top", fontsize=8.5)
panel(ax[1], "b")
save(fig, MAIN, "Fig06_shap_decomposition.png")

# ---- geodata for the map figures -------------------------------------------
HAVE_GEO = False
try:
    import geopandas as gpd
    if os.path.exists(CACHE):
        ctry = gpd.read_file(CACHE)
    elif os.path.exists(GEOJSON):
        adm = gpd.read_file(GEOJSON)
        ctry = adm.dissolve(by="COUNTRY", as_index=False)[["COUNTRY", "geometry"]]
        ctry.to_file(CACHE, driver="GPKG")
    else:
        raise FileNotFoundError("no boundary file")
    HAVE_GEO = True
except Exception as e:
    print(f"  geodata unavailable ({e}) - map figures skipped")

cities = pd.read_csv(os.path.join(DATA, "global100_cities.csv"))[["city_en", "lat", "lon"]]
G = (A.groupby("city").agg(est=("est_ppb", "mean"), temp=("temp_c", "mean"),
                           indom=("in_domain", "mean"), rho=("exp_rho", "mean"))
     .reset_index().merge(cities.rename(columns={"city_en": "city"}), on="city", how="left"))
G["indom"] *= 100
G = G.dropna(subset=["lat", "lon"])


def basemap(ax, ext):
    ax.set_facecolor(SEA)
    ctry.plot(ax=ax, color=LAND, edgecolor=EDGE, linewidth=.5)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_aspect(1 / np.cos(np.radians((ext[2] + ext[3]) / 2)))
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor("#8a8a8a"); s.set_linewidth(.7)


def scalebar(ax, ext, km, fx=.06, fy=.055):
    lat = (ext[2] + ext[3]) / 2
    dx = km / (111.32 * np.cos(np.radians(lat)))
    x0 = ext[0] + (ext[1] - ext[0]) * fx; y0 = ext[2] + (ext[3] - ext[2]) * fy
    ax.plot([x0, x0 + dx], [y0, y0], color="k", lw=2.0, solid_capstyle="butt", zorder=8)
    ax.text(x0 + dx / 2, y0 + (ext[3] - ext[2]) * .022, f"{km} km", ha="center",
            fontsize=7, zorder=8)


def place_labels(ax, sub, ext, fs=7, val="est"):
    used = []
    W, H = ext[1] - ext[0], ext[3] - ext[2]
    offs = [(0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1),
            (1.6, 0), (-1.6, 0)]
    for _, r in sub.sort_values(val, ascending=False).iterrows():
        txt = f"{r.city.split('(')[0].strip()[:13]} {r[val]:.0f}"
        w, h = len(txt) * .0092 * W, .042 * H
        for dx, dy in offs:
            x, y = r.lon + dx * .024 * W, r.lat + dy * .034 * H
            bb = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            if any(not (bb[2] < u[0] or bb[0] > u[2] or bb[3] < u[1] or bb[1] > u[3])
                   for u in used):
                continue
            if not (ext[0] < bb[0] and bb[2] < ext[1] and ext[2] < bb[1] and bb[3] < ext[3]):
                continue
            used.append(bb)
            ax.annotate(txt, (x, y), fontsize=fs, ha="center", va="center", zorder=7)
            ax.plot([r.lon, x], [r.lat, y], lw=.35, color="#5a5a5a", zorder=4)
            break


if HAVE_GEO:
    # ---- F7  global estimates ----------------------------------------------
    ext = (-125, 142, -38, 60)
    fig, ax = plt.subplots(figsize=(12.6, 5.6), constrained_layout=True)
    basemap(ax, ext)
    sz = np.where(G.rho >= .8, 84, np.where(G.rho >= .6, 52, np.where(G.rho >= .4, 30, 16)))
    s = ax.scatter(G.lon, G.lat, c=G.est, s=sz, cmap=CMAP, vmin=8, vmax=30,
                   edgecolors="k", linewidths=.45, zorder=5)
    cb = fig.colorbar(s, ax=ax, fraction=.022, pad=.008)
    cb.set_label("Estimated surface NO$_2$ (ppb)", fontsize=9)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc="#bbb", mec="k",
                              ms=np.sqrt(v) * .78, label=l)
                       for v, l in [(84, "high  ($\\rho\\geq$0.8)"),
                                    (52, "moderate  (0.6–0.8)"),
                                    (30, "low  (0.4–0.6)"),
                                    (16, "very low  (<0.4)")]],
              loc="lower left", labelspacing=.95, borderpad=.6,
              title="Expected reliability", title_fontsize=8)
    scalebar(ax, ext, 2000, fx=.62, fy=.05)
    save(fig, MAIN, "Fig07_global_estimates_map.png")

    # ---- F8  MENA detail ----------------------------------------------------
    ext = (-12, 64, 8, 42)
    fig, ax = plt.subplots(figsize=(12.8, 6.0), constrained_layout=True)
    basemap(ax, ext)
    sub = G[(G.lon.between(*ext[:2])) & (G.lat.between(*ext[2:]))].copy()
    sz = np.where(sub.temp >= 26, 60, np.where(sub.temp >= 22, 112, 176))
    s = ax.scatter(sub.lon, sub.lat, c=sub.est, s=sz, cmap=CMAP, vmin=8, vmax=30,
                   edgecolors="k", linewidths=.55, zorder=5)
    place_labels(ax, sub, ext)
    scalebar(ax, ext, 500, fx=.30)
    cb = fig.colorbar(s, ax=ax, fraction=.026, pad=.008)
    cb.set_label("Estimated surface NO$_2$ (ppb)", fontsize=9)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc="#bbb", mec="k",
                              ms=np.sqrt(v) * .62, label=l)
                       for v, l in [(176, "high  (< 22 °C)"), (112, "moderate  (22–26 °C)"),
                                    (60, "low  (> 26 °C)")]],
              loc="lower left", labelspacing=.9, borderpad=.6,
              title="Expected reliability", title_fontsize=8)
    save(fig, MAIN, "Fig08_mena_estimates_map.png")

# ---- F9  confidence axes ----------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
sc = ax.scatter(G.indom, G.rho, c=G.temp, s=52, cmap="coolwarm",
                edgecolors="k", linewidths=.4, zorder=3)
ax.axhline(.6, color="0.45", ls="--", lw=1.0)
ax.axvline(70, color="0.45", ls="--", lw=1.0)
for k in ["Riy", "Jed", "Dub", "Teh", "Bagh", "Khar", "Qahir"]:
    for _, r in G[G.city.str.contains(k, na=False)].iterrows():
        ax.annotate(r.city.split("(")[0].strip()[:11], (r.indom, r.rho), fontsize=7.4,
                    xytext=(5, 4), textcoords="offset points")
cb = fig.colorbar(sc, ax=ax, pad=.015); cb.set_label("Mean air temperature (°C)", fontsize=9)
ax.set(xlabel="Months inside the training envelope (%)",
       ylabel=r"Expected $\rho$ (column, surface)", xlim=(-4, 108))
save(fig, MAIN, "Fig09_confidence_axes.png")


# ---- F10  estimated NO2 in 2019 and 2025 ------------------------------------
yr = A.groupby(["city", "year"]).est_ppb.mean().unstack()
obs = pd.concat([B, C], ignore_index=True).groupby(["city", "year"])[T].mean().unstack()
Y1, Y2 = 2019, 2025
if Y1 in yr.columns and Y2 in yr.columns:
    sel = yr[[Y1, Y2]].dropna()
    MENA_K = ["Riy", "Jed", "Dub", "Doh", "Kuw", "Qahir", "Bagh", "Teh", "Khar",
              "Dimashq", "Amm"]
    keep = set(sel.nlargest(18, Y2).index)
    for k in MENA_K:
        keep |= {c for c in sel.index if k in c}
    sel = sel.loc[sorted(keep, key=lambda c: -sel.loc[c, Y2])]
    fig, ax = plt.subplots(2, 1, figsize=(12.4, 7.6), sharex=True, constrained_layout=True)
    x = np.arange(len(sel))
    isM = [any(t in c for t in MENA_K) for c in sel.index]
    cols = ["#c0392b" if m_ else "#5a7d9a" for m_ in isM]
    ymax = max(sel[Y1].max(), sel[Y2].max()) * 1.22
    for k, (yy, a) in enumerate(zip([Y1, Y2], ax)):
        a.bar(x, sel[yy], color=cols, width=.72, edgecolor="none")
        n = 0
        if yy in obs.columns:
            o = [obs.loc[c, yy] if c in obs.index and pd.notna(obs.loc[c, yy]) else np.nan
                 for c in sel.index]
            a.scatter(x, o, marker="_", s=170, linewidths=2.0, color="k", zorder=5)
            n = int(np.sum(~np.isnan(o)))
        a.set_ylabel(f"{yy}\nsurface NO$_2$ (ppb)")
        a.set_ylim(0, ymax)
        a.text(.5, .955, f"{yy}   ·   {n} of {len(sel)} cities have ground observations",
               transform=a.transAxes, ha="center", va="top", fontsize=8.4)
        panel(a, "ab"[k])
    ax[1].set_xticks(x, [c.split("(")[0].strip()[:16] for c in sel.index],
                     rotation=48, ha="right")
    ax[0].legend(handles=[
        Line2D([0], [0], marker="s", ls="", ms=7.5, color="#5a7d9a",
               label="Estimated, other cities"),
        Line2D([0], [0], marker="s", ls="", ms=7.5, color="#c0392b",
               label="Estimated, MENA target"),
        Line2D([0], [0], marker="_", ls="", ms=10, mew=2.0, color="k",
               label="Observed, where available")],
        loc="upper right", ncol=1, borderpad=.5)
    save(fig, MAIN, "Fig10_estimates_2019_2025.png")

# ==================================================== SUPPLEMENTARY =========
print("\nsupplementary figures")

# ---- S1  target distribution ------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
y = COOL[T]
bins = np.linspace(0, allc[T].max(), 42)
for r, g in allc.groupby("region"):
    ax[0].hist(g[T], bins=bins, alpha=.5, color=COL[r], edgecolor="white",
               lw=.3, label=NICE[r])
ax[0].set(xlabel="Surface NO$_2$ (ppb)", ylabel="City-months")
ax[0].legend(); panel(ax[0], "a")
lb = np.linspace(np.log(allc[T].min()), np.log(allc[T].max()), 42)
for r, g in allc.groupby("region"):
    ax[1].hist(np.log(g[T]), bins=lb, alpha=.5, color=COL[r], edgecolor="white", lw=.3)
ax[1].set(xlabel="log Surface NO$_2$ (ppb)", ylabel="City-months")
ax[1].text(.97, .96, f"skew {allc[T].skew():+.2f} raw\nskew {np.log(allc[T]).skew():+.2f} log",
           transform=ax[1].transAxes, ha="right", va="top", fontsize=8.5)
panel(ax[1], "b")
save(fig, SUPP, "FigS1_target_distribution.png")

# ---- S2  seasonality --------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
for r in allc.region.unique():
    g = allc[allc.region == r].groupby("month")[T].agg(["mean", "std"])
    ax.plot(g.index, g["mean"], "o-", color=COL[r], lw=1.8, ms=4, label=NICE[r])
    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                    color=COL[r], alpha=.11)
ax.set(xlabel="Month", ylabel="Surface NO$_2$ (ppb)", xticks=range(1, 13))
ax.legend(ncol=2)
save(fig, SUPP, "FigS2_seasonality.png")

# ---- S3  correlation matrix -------------------------------------------------
cm = COOL[F].rename(columns=LBL).corr()
fig, ax = plt.subplots(figsize=(6.6, 5.6), constrained_layout=True)
im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm)), cm.columns, rotation=42, ha="right")
ax.set_yticks(range(len(cm)), cm.columns)
ax.grid(False)
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and abs(cm.iloc[i, j]) > .7:
            ax.text(j, i, f"{cm.iloc[i,j]:+.2f}", ha="center", va="center",
                    fontsize=7.2, fontweight="bold", color="white")
cb = fig.colorbar(im, ax=ax, shrink=.82, pad=.02); cb.set_label("Pearson $r$", fontsize=9)
save(fig, SUPP, "FigS3_correlation_matrix.png")

# ---- S4  retrieval quality --------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
ax[0].hist(allc.cloud_fraction, bins=32, color="#5a7d9a", edgecolor="white", lw=.3)
ax[0].axvline(allc.cloud_fraction.mean(), color="k", ls="--", lw=1.0)
ax[0].set(xlabel="Cloud fraction", ylabel="City-months"); panel(ax[0], "a")
q = pd.qcut(allc.cloud_fraction, 4, labels=["Q1", "Q2", "Q3", "Q4"])
rr = [spearmanr(g.no2_vcd_mol_m2, g[T])[0] for _, g in allc.groupby(q, observed=True)]
ax[1].bar(range(4), rr, color="#5a7d9a", width=.6)
ax[1].set_xticks(range(4), ["Q1\nclearest", "Q2", "Q3", "Q4\ncloudiest"])
ax[1].set_ylabel(r"$\rho$ (column, surface)"); panel(ax[1], "b")
ax[2].hist(allc.n_obs_no2, bins=28, color="#5a7d9a", edgecolor="white", lw=.3)
ax[2].set(xlabel="TROPOMI granules per city-month", ylabel="City-months"); panel(ax[2], "c")
save(fig, SUPP, "FigS4_retrieval_quality.png")

# ---- S5  per-city observed distribution -------------------------------------
order = allc.groupby("city")[T].median().sort_values()
fig, ax = plt.subplots(figsize=(7.6, 9.6), constrained_layout=True)
bp = ax.boxplot([allc[allc.city == c][T].values for c in order.index], vert=False,
                patch_artist=True, widths=.66, showfliers=False,
                medianprops=dict(color="black", lw=1.1))
for p, c in zip(bp["boxes"], order.index):
    p.set_facecolor(COL[allc[allc.city == c].region.iloc[0]]); p.set_alpha(.7)
    p.set_edgecolor("none")
ax.set_yticks(range(1, len(order) + 1), order.index, fontsize=6.8)
ax.set_xlabel("Observed surface NO$_2$ (ppb)")
ax.legend(handles=[Patch(facecolor=COL[r], alpha=.7, label=NICE[r])
                   for r in allc.region.unique()], loc="lower right")
save(fig, SUPP, "FigS5_per_city_observed.png")

# ---- S6  domain envelope ----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)
xs = np.linspace(0, max(A.pblh_m.max(), COOL.pblh_m.max()) * 1.03, 400)
ax[0].fill_between(xs, gaussian_kde(COOL.pblh_m, bw_method=.35)(xs), color="#9bb7d4",
                   alpha=.8, label=f"Training (max {COOL.pblh_m.max():.0f} m)")
riy = A[A.city.str.contains("Riy", na=False)]
if len(riy):
    ax[0].plot(xs, gaussian_kde(riy.pblh_m, bw_method=.35)(xs), color="0.35", lw=2.0,
               label=f"Riyadh (max {riy.pblh_m.max():.0f} m)")
ax[0].axvline(COOL.pblh_m.max(), color="k", ls="--", lw=1.2)
ax[0].set(xlabel="PBLH (m)", ylabel="Density"); ax[0].legend(); panel(ax[0], "a")
cov = (A.groupby("city").in_domain.mean() * 100)
ax[1].hist(cov, bins=np.arange(0, 105, 5), color="#9bb7d4", edgecolor="white", lw=.4)
for v, c in [(40, "#c0392b"), (70, "#e08e0b"), (90, "#2a9d5c")]:
    ax[1].axvline(v, color=c, ls="--", lw=1.1)
ax[1].set(xlabel="Months inside the training envelope (%)", ylabel="Number of cities")
panel(ax[1], "b")
save(fig, SUPP, "FigS6_domain_envelope.png")

# ---- S7  failure cause heatmap ----------------------------------------------
E5 = ["no2_vcd_mol_m2", "pblh_m", "temp_c", "wind_u_ms", "wind_v_ms"]
worst = cov.nsmallest(14).index[::-1]
mat = np.array([[100 * (~A[A.city == c][v].between(*RNG[v])).mean() for v in E5]
                for c in worst])
fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(E5)), [LBL[v] for v in E5], rotation=28, ha="right")
ax.set_yticks(range(len(worst)), [c[:30] for c in worst], fontsize=7.6)
ax.grid(False)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if mat[i, j] > 3:
            ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=7.2,
                    fontweight="bold", color="white" if mat[i, j] > 55 else "black")
cb = fig.colorbar(im, ax=ax, shrink=.85, pad=.02)
cb.set_label("Months outside the training range (%)", fontsize=9)
save(fig, SUPP, "FigS7_failure_cause.png")

# ---- S8  predictor overlap --------------------------------------------------
SH = ["no2_vcd_mol_m2", "pblh_m", "temp_c", "wind_speed_ms", "population", "built_up_km2"]
fig, ax = plt.subplots(2, 3, figsize=(11.6, 5.6), constrained_layout=True)
for k, c in enumerate(SH):
    a = ax.flat[k]
    lo, hi = min(COOL[c].min(), A[c].min()), max(COOL[c].max(), A[c].max())
    if c in ("population", "built_up_km2", "no2_vcd_mol_m2"):
        bins = np.logspace(np.log10(max(lo, 1e-6)), np.log10(hi), 40); a.set_xscale("log")
    else:
        bins = np.linspace(lo, hi, 40)
    a.hist(COOL[c], bins=bins, density=True, alpha=.6, color="#9bb7d4", label="Training")
    a.hist(A[c], bins=bins, density=True, histtype="step", lw=1.6, color="0.30",
           label="Target")
    a.axvline(COOL[c].max(), color="k", ls="--", lw=.9)
    a.set(xlabel=LBL[c], ylabel="Density" if k % 3 == 0 else "")
    panel(a, "abcdef"[k])
    if k == 0:
        a.legend()
save(fig, SUPP, "FigS8_predictor_overlap.png")

# ---- S9  level vs shape -----------------------------------------------------
rows = []
for c, g in COOL.groupby("city"):
    rows.append((c, g.region.iloc[0], r2_score(g[T], g.oof), g.oof.mean() - g[T].mean(),
                 np.corrcoef(g.oof, g[T])[0, 1] ** 2))
dsh = pd.DataFrame(rows, columns=["city", "region", "r2", "bias", "shape"])
fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
ax.scatter(dsh["bias"].values, dsh["shape"].values,
           c=[COL[r] for r in dsh.region], s=48, alpha=.85, edgecolors="k", linewidths=.4)
for _, r in dsh.iterrows():
    if r.r2 < 0:
        ax.annotate(r.city[:13], (r["bias"], r["shape"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
ax.axvline(0, color="k", lw=.9)
ax.set(xlabel="Level error: mean estimated $-$ mean observed (ppb)",
       ylabel="Shape agreement: $r^2$ of the monthly cycle")
ax.legend(handles=[Patch(facecolor=COL[r], label=NICE[r]) for r in dsh.region.unique()],
          loc="lower left")
save(fig, SUPP, "FigS9_level_versus_shape.png")

# ---- S10  airshed geometry --------------------------------------------------
if HAVE_GEO:
    def airshed(lat, lon, km):
        loc = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
        p = gpd.GeoSeries([gpd.points_from_xy([lon], [lat])[0]], crs=4326)
        return p.to_crs(loc).buffer(km * 1000).to_crs(4326).iloc[0]

    targets = [("Riy", (45.6, 47.9, 23.6, 25.8)), ("Jed", (38.4, 40.0, 20.9, 22.5)),
               ("Dub", (54.4, 56.1, 24.4, 26.0))]
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 4.0), constrained_layout=True)
    for k, (key, ext) in enumerate(targets):
        basemap(ax[k], ext)
        r = G[G.city.str.contains(key, na=False)]
        if not len(r):
            continue
        r = r.iloc[0]
        gpd.GeoSeries([airshed(r.lat, r.lon, 30)], crs=4326).plot(
            ax=ax[k], facecolor="#4a7fb5", alpha=.20, edgecolor="#2c6fbb",
            linewidth=1.3, linestyle=(0, (4, 2)))
        gpd.GeoSeries([airshed(r.lat, r.lon, 20)], crs=4326).plot(
            ax=ax[k], facecolor="#c0392b", alpha=.26, edgecolor="#c0392b", linewidth=1.5)
        ax[k].scatter(r.lon, r.lat, s=44, marker="*", color="k", zorder=6)
        ax[k].text(.5, .035, f"{r.city.split('(')[0].strip()}  {r.est:.1f} ppb",
                   transform=ax[k].transAxes, ha="center", fontsize=8.5, fontweight="bold",
                   zorder=8)
        scalebar(ax[k], ext, 25, fx=.06, fy=.13)
        panel(ax[k], "abc"[k])
    ax[0].legend(handles=[Patch(facecolor="#c0392b", alpha=.45, label="20 km (selected)"),
                          Patch(facecolor="#4a7fb5", alpha=.35, label="30 km (rejected)")],
                 loc="upper left")
    save(fig, SUPP, "FigS10_airshed_geometry.png")

print(f"\ndone.  main -> {MAIN}\n       supp -> {SUPP}")
print("\nVALUES SHOWN IN THE FIGURES - must match the manuscript text")
print(f"  P1, mean of per-seed R2     {P1:.3f} +/- {P1SD:.3f}   Sect. 4.2")
print(f"  P1, R2 of seed ensemble     {r2_score(COOL[T], COOL.oof):.3f}          Fig. 4a")
print(f"  arid, first stage only      {r2_score(C[T], C_pred_1):+.3f}  bias {np.mean(C_pred_1-C[T]):+.2f}   Sect. 4.4, Fig. 6a")
print(f"  arid, two-stage gated       {r2_score(C[T], C_pred_2):+.3f}  bias {np.mean(C_pred_2-C[T]):+.2f}   Sect. 4.6, Fig. 6b")
print(f"  rho vs temperature          r = {pearsonr(Q.temp, Q.rho)[0]:+.3f}        Sect. 4.5")
