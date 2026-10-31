# Figure Captions

All figures 720 dpi, no on-figure titles, no overlapping elements. Numbering is
in the filenames only. Every value is computed live by `make_all_figures.py`
from the three frozen datasets, so captions and figures cannot drift apart.

**Datasets.** A — 100 target cities, no ground truth. B — 49 training cities
(15 US EPA, 20 China CNEMC, 10 India CPCB, 4 Gulf ADQCC). C — 5 arid
north-west Indian cities, held out and opened once.

**Model.** Stage 1: XGBoost on the 35 strong-link cities (US + China). Stage 2:
XGBoost on the hot/arid cities, predicting stage 1's residual, **climate-gated**
so that it is applied only where it was fitted: w = clip((T_city − 23)/(25 − 23),
0, 1), evaluated at each city's long-run mean temperature. Without the gate,
stage 2 inflates temperate cities by roughly 6 ppb. All deployment predictions
are the **mean over 10 model seeds**; a single fit is one draw from XGBoost's
subsampling distribution and out-of-domain cities are the most sensitive to it
(Riyadh moved 2 ppb between two runs of identical code). All predictors averaged
over a 20 km airshed; concentrations in ppb (× 1.88 for µg/m³).

---

## Main text

**Figure 0.** Analysis workflow. Five stages run top to bottom: acquisition of
satellite, meteorological, urban and ground-network data; harmonisation over a
20 km airshed under a single enforced unit contract; assembly of three frozen,
leakage-checked datasets; modelling under a protocol fixed before any fitting;
and a confidence-stratified product. The held-out arid test set (purple) is
routed separately to show it is opened once, at the end. Counts state the data
volume at each stage, so the figure doubles as a provenance summary.

**Figure 1.** Relationship between the TROPOMI tropospheric NO₂ column and
observed surface NO₂, by region. (a) Monthly city means with ordinary
least-squares fits, one per region. (b) Ratio of mean surface concentration to
mean column. The conversion differs roughly threefold between regions, which is
the direct cause of the cross-region transfer failure shown in Figure 5.

**Figure 2.** Surface NO₂ anomaly against boundary-layer-height anomaly, both
relative to each city's long-run mean, by region, with least-squares fits.
Removing city means isolates the within-city relationship, in which a deeper
mixing layer is associated with lower surface concentration in every region —
the boundary-layer "lid" effect. The relationship is not visible in raw levels
because it is masked by between-city differences in emission intensity.

**Figure 3.** Performance of the stage-1 inversion under leave-one-city-out
validation on the 35 strong-link cities. (a) Estimated against observed monthly
surface NO₂; the dashed line is 1:1. (b) Residuals against estimate, with decile
means in red, showing no systematic curvature. (c) Per-city R², ordered, with
error bars giving the standard deviation across model seeds. Every estimate
comes from a model that never saw that city.

**Figure 4.** Strength of the column-to-surface relationship, measured as the
Spearman correlation between the satellite column and observed surface NO₂
within each city, against (a) mean air temperature and (b) mean boundary-layer
height, for all 54 cities with ground truth. The dashed line in (a) is the
fitted relationship ρ = tanh(a·T + b); in (b) it is an ordinary least-squares
fit. The vertical guide marks Riyadh's mean temperature and the corresponding
expected correlation. The systematic decline with temperature is the central
result of this study: the physical signal the inversion depends upon is weaker
in hot climates, independently of model choice or training-set size.

**Figure 5.** Transfer to the held-out arid test set — five north-west Indian
cities never used in fitting, tuning or feature selection. (a) Single-stage
model trained on the strong-link regions only. (b) Two-stage model with the arid
residual correction. (c) Mean bias by city under both designs. The correction
removes almost all of the systematic under-prediction while recovering little
variance, so arid estimates are approximately unbiased but imprecise.

**Figure 6.** SHAP attribution for the stage-1 model, computed with XGBoost's
native TreeSHAP. (a) Variance of each feature's SHAP values, split into a
between-city component, which sets a city's baseline level, and a within-city
component, which tracks month-to-month variation. Percentages give the
within-city share. Built-up area and population act almost entirely between
cities; the NO₂ column is the only predictor contributing substantially to both;
PBLH is almost purely temporal. (b) SHAP value for PBLH against PBLH, coloured
by temperature; the negative slope is the lid effect. Mean |SHAP| alone would
rank built-up area first and obscure this structure.

**Figure 7.** Estimated mean surface NO₂ for the 100 target cities, 2019–2025.
Marker size encodes expected reliability from the ρ–temperature relationship of
Figure 4. Estimates are shown for every city; those with small markers should be
read with the corresponding uncertainty.

**Figure 8.** Detail of Figure 7 for the Middle East, North Africa and southern
Europe, with per-city estimates in ppb. Marker size encodes expected reliability
by mean temperature. All Gulf targets fall in the lowest reliability class.

**Figure 9.** The two confidence measures reported for every target city, plotted
against each other and coloured by mean temperature. The horizontal axis gives
the share of months whose predictors lie inside the stage-1 training envelope
(feature-space extrapolation); the vertical axis gives the expected
column-to-surface correlation (physical signal strength). The measures are
independent: Jeddah and Dubai are almost fully in-domain yet have weak expected
correlation, whereas Riyadh is partly out-of-domain. Reporting either alone
would misrepresent these cities in opposite directions.

**Figure 10.** Estimated mean surface NO₂ in (a) 2019 and (b) 2025 for the 18
highest-concentration target cities together with all MENA targets, ordered by
the 2025 estimate and shown in the same order in both panels. MENA targets are
red. Black horizontal marks give the observed value where ground data exist:
one city in 2019 and four in 2025, since the Chinese, Indian and Gulf networks
used here begin in 2022 or later. The scarcity of observations in 2019 is itself
informative — it is the data gap the method is intended to fill.

---

## Supplementary

**Figure S1.** Distribution of monthly mean surface NO₂ across all cities with
ground truth, on (a) the raw and (b) the natural-log scale, by region.

**Figure S2.** Mean monthly cycle of observed surface NO₂ by region; shading
gives ±1 standard deviation.

**Figure S3.** Pearson correlation matrix of model predictors over the
strong-link training cities. Pairs exceeding |r| = 0.7 are labelled.

**Figure S4.** Retrieval quality. (a) Distribution of monthly mean cloud
fraction; the dashed line is the mean. (b) Spearman correlation between column
and surface concentration within cloud-fraction quartiles, showing no
degradation with increasing cloud. (c) Number of TROPOMI granules contributing
to each city-month.

**Figure S5.** Distribution of observed monthly surface NO₂ for every city with
ground truth, ordered by median and coloured by region. Whiskers extend to
1.5 × IQR; outliers are suppressed for legibility.

**Figure S6.** (a) Kernel density of boundary-layer height in the training set
and at Riyadh; the dashed line marks the training maximum. (b) Distribution
across the 100 target cities of the share of months lying inside the training
envelope.

**Figure S7.** For the 14 least-supported target cities, the share of months in
which each predictor falls outside the training range. Two distinct failure
modes are visible: equatorial African cities lie below the training minimum for
the NO₂ column, whereas interior Arabian cities lie above the maximum for
boundary-layer height.

**Figure S8.** Overlap of predictor distributions between the strong-link
training set (filled) and the target set (outline). The dashed line marks the
training maximum. Column, population and built-up area are on logarithmic axes.

**Figure S9.** Decomposition of per-city error into a level component (mean
estimated minus mean observed) and a shape component (r² of the monthly cycle),
for the strong-link training cities under leave-one-city-out validation. Cities
with negative R² are labelled. Failures are predominantly level errors: the
model reproduces the seasonal cycle but places the city at the wrong baseline.

**Figure S10.** Airshed geometry over Riyadh, Jeddah and Dubai. The 20 km airshed
(red, selected by cross-validation) and the 30 km airshed (blue, tested and
rejected) are drawn at true scale by buffering in a local azimuthal equidistant
projection centred on each city. All satellite, meteorological and urban
predictors were averaged over the airshed; municipal boundaries were not used at
any stage of the analysis.

---

## Content presented as tables, not figures

Each of the following is a small set of numbers that a table conveys more
precisely and compactly than a chart:

| Content | Values | Destination |
|---|---|---|
| Per-feature Spearman correlation with surface NO₂, by region | 10 × 2 | Table |
| City-months by region and year | 5 × 7 | Table |
| Standardised mean difference, target versus training | 6 | Table |
| Feature-block ablation (P1, within-city, between-city) | 4 × 3 | Table |
| Design ablation on the held-out arid test | 5 | Table |
| Per-city skill with seed spread | 50 rows | Supplementary table |
| Estimates for all 100 cities with both confidence measures | 100 rows | Supplementary table |
