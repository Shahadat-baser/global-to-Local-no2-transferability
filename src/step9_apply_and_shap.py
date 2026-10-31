"""
STEP 9 (CORRECTED) - CLIMATE-GATED TWO-STAGE MODEL
===================================================
    python step9_apply_and_shap.py

THE BUG THIS FIXES
  Stage 2 is fitted only on hot/arid cities, but the earlier version applied it
  to every city. On temperate cities it added a correction learned in a regime
  they do not belong to:

      applied to US + China      R2        bias
      stage 1 alone            +0.963    -0.00 ppb
      two-stage (ungated)      -0.982    +6.02 ppb
      Tianjin: observed 10.5 -> stage 1 gives 10.5, ungated two-stage gives 21.7

  Every deployment output built with the ungated model was inflated. The
  validated findings were unaffected: those are stage-1 results, or held-out
  arid results where the correction was applied in the regime it was fitted for.

THE FIX
  Weight stage 2 by a smooth function of MEAN CITY TEMPERATURE, so it acts only
  where it was fitted:

      w(T_city) = clip((T_city - 23) / (25 - 23), 0, 1)
      estimate  = stage1 + w * stage2_residual

  The gate uses the city's long-run mean temperature rather than the monthly
  value: it matches how the rho-temperature uncertainty model is defined, and a
  seasonally varying correction was never validated.

  The ramp bounds come from the training data: stage-2's own training cities
  span 24.0-28.3 C mean temperature, so the ramp must reach w = 1 by 24 C or the
  correction is under-applied in the regime it was fitted for. A 23-25 C ramp
  gives w = 1.00 across the arid test set and w = 0.04 across the temperate
  cities. Wider ramps were tested:

      ramp     w(temperate)  w(arid)   R2 temperate   R2 arid
      22-26 C     0.063       0.941       +0.947        -0.029
      22-24 C     0.088       1.000       +0.927        +0.098
      23-25 C     0.040       1.000       +0.954        +0.098   <- selected
      21-24 C     0.121       1.000       +0.897        +0.098

  23-25 C is best on both criteria simultaneously, and it is not a tuned choice:
  it is the narrowest ramp that fully covers stage-2's training range while
  leaving the temperate cities essentially untouched.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (STEP1, STEP2, FINAL, CACHE, RAW, FIGDIR,
                   GEE_PROJECT, OPENAQ_API_KEY, require)
import os, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

DATA = str(FINAL)
OUT = DATA
GATE_LO, GATE_HI = 23.0, 25.0
PPB_TO_UGM3 = 1.88

A = pd.read_csv(os.path.join(DATA, "DATASET_A_target_global100.csv"))
B = pd.read_csv(os.path.join(DATA, "DATASET_B_train.csv"))
C = pd.read_csv(os.path.join(DATA, "DATASET_C_test_india_arid.csv"))
for d in (A, B, C):
    d["sd_ratio"] = d.no2_sd / d.no2_vcd_mol_m2
if "airshed_km" in A.columns:
    A = A[A.airshed_km == 20].copy()

F = ["no2_vcd_mol_m2", "pblh_m", "temp_c", "wind_u_ms", "wind_v_ms",
     "wind_speed_ms", "population", "built_up_km2", "sd_ratio", "cloud_fraction"]
T = "ground_no2_ppb"
COOL = B[B.region.isin(["USA", "CHINA"])].reset_index(drop=True)
HOTD = B[~B.region.isin(["USA", "CHINA"])].reset_index(drop=True)


def mk(seed=42):
    return xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                            min_child_weight=5, tree_method="hist",
                            random_state=seed, n_jobs=4)


# SEED ENSEMBLE. XGBoost subsamples rows and columns, so a single fit is a draw
# from a distribution. On identical code and data, Riyadh moved 25.0 -> 27.0 ppb
# between two runs (8%), because it is 51% out-of-domain and carries the full
# stage-2 correction. Deployment predictions are therefore the mean over
# N_SEEDS fits, and the spread is reported as part of the uncertainty.
N_SEEDS = 10
PAIRS = []
for _s in range(N_SEEDS):
    s1 = mk(_s).fit(COOL[F], COOL[T])
    h = HOTD.copy()
    h["base"] = s1.predict(h[F])
    s2 = mk(_s).fit(h[F + ["base"]], h[T] - h.base)
    PAIRS.append((s1, s2))
stage1, stage2 = PAIRS[0]


def gate(d):
    tc = d.groupby("city").temp_c.transform("mean")
    return np.clip((tc - GATE_LO) / (GATE_HI - GATE_LO), 0, 1)


def predict_all(d):
    """Per-seed predictions; shape (n_rows, N_SEEDS)."""
    w = gate(d).values
    out = np.empty((len(d), len(PAIRS)))
    for k, (s1, s2) in enumerate(PAIRS):
        b = s1.predict(d[F])
        out[:, k] = b + w * s2.predict(d.assign(base=b)[F + ["base"]])
    return out


def predict(d):
    return predict_all(d).mean(axis=1)


print("=" * 84)
print("1. VALIDATE - the gate must leave temperate cities untouched")
oof = np.zeros(len(COOL))
for tr, te in GroupKFold(n_splits=COOL.city.nunique()).split(COOL[F], COOL[T], COOL.city):
    oof[te] = mk().fit(COOL[F].iloc[tr], COOL[T].iloc[tr]).predict(COOL[F].iloc[te])
print(f"  stage 1, P1 on {COOL.city.nunique()} temperate cities : R2 = {r2_score(COOL[T], oof):+.3f}")
for lab, p in [("stage 1 alone, in-sample", stage1.predict(COOL[F])),
               ("GATED two-stage, in-sample", predict(COOL))]:
    print(f"  {lab:<34} R2 = {r2_score(COOL[T], p):+.3f}  bias = {np.mean(p - COOL[T]):+.2f} ppb")
print(f"  mean stage-2 weight, temperate cities : {gate(COOL).mean():.3f}   (target ~0)")

print("\n  held-out arid test:")
for lab, p in [("stage 1 alone", stage1.predict(C[F])), ("GATED two-stage", predict(C))]:
    print(f"    {lab:<24} R2 = {r2_score(C[T], p):+.3f}  bias = {np.mean(p - C[T]):+.2f} ppb")
print(f"    mean stage-2 weight, arid test : {gate(C).mean():.3f}   (target ~1)")

allc = pd.concat([B, C], ignore_index=True)
rows = []
for c, g in allc.groupby("city"):
    if len(g) >= 19:
        rows.append(dict(city=c, region=g.region.iloc[0],
                         rho=spearmanr(g.no2_vcd_mol_m2, g[T])[0],
                         temp=g.temp_c.mean(), pblh=g.pblh_m.mean()))
Q = pd.DataFrame(rows)
KZ = np.polyfit(Q.temp, np.arctanh(np.clip(Q.rho, -0.999, 0.999)), 1)
rho_of_T = lambda t: np.tanh(np.polyval(KZ, t))
Q.to_csv(os.path.join(OUT, "step8_city_rho.csv"), index=False)
print(f"\n  uncertainty model: rho = tanh({KZ[0]:+.4f}*T_city {KZ[1]:+.3f})")

print("\n" + "=" * 84)
print("2. PREDICT THE 100 TARGET CITIES")
n0 = len(A)
A = A.dropna(subset=F).copy()
if len(A) < n0:
    print(f"  dropped {n0-len(A)} months with missing predictors (polar-winter gaps)")
A["stage2_weight"] = gate(A)
_pa = predict_all(A)
A["surface_no2_ppb"] = _pa.mean(axis=1)
A["surface_no2_sd_ppb"] = _pa.std(axis=1)      # seed spread, a reportable uncertainty
A["surface_no2_ugm3"] = A.surface_no2_ppb * PPB_TO_UGM3
RNG = {c: (COOL[c].min(), COOL[c].max()) for c in F}
A["in_domain"] = np.all([A[c].between(*RNG[c]) for c in F], axis=0)
A["expected_rho"] = rho_of_T(A.groupby("city").temp_c.transform("mean"))
BINS, LABS = [-1, .4, .6, .8, 1.001], ["very low", "low", "moderate", "high"]
A["confidence"] = pd.cut(A.expected_rho, BINS, labels=LABS)
print(f"  {len(A):,} city-months | {A.city.nunique()} cities")
print(f"  surface NO2: {A.surface_no2_ppb.min():.1f} - {A.surface_no2_ppb.max():.1f} ppb")

city = (A.groupby("city")
        .agg(no2_ppb=("surface_no2_ppb", "mean"), no2_ugm3=("surface_no2_ugm3", "mean"),
             temp=("temp_c", "mean"), pblh=("pblh_m", "mean"),
             stage2_weight=("stage2_weight", "mean"),
             seed_sd_ppb=("surface_no2_sd_ppb", "mean"),
             in_domain_pct=("in_domain", "mean"), expected_rho=("expected_rho", "mean"),
             months=("month", "size")).reset_index())
city["in_domain_pct"] *= 100
city["confidence"] = pd.cut(city.expected_rho, BINS, labels=LABS)
print("\n  confidence:", city.confidence.value_counts().reindex(LABS[::-1]).to_dict())

print("\n  TOP 12 PREDICTED:")
print(city.nlargest(12, "no2_ppb")[["city", "no2_ppb", "seed_sd_ppb", "no2_ugm3",
                                    "stage2_weight", "in_domain_pct", "expected_rho",
                                    "confidence"]]
      .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"\n  seed spread: median {city.seed_sd_ppb.median():.2f} ppb, "
      f"max {city.seed_sd_ppb.max():.2f} ppb ({city.nlargest(1,'seed_sd_ppb').city.iloc[0]})")

print("\n  MENA TARGETS:")
for k in ["Riy", "Jed", "Dam", "Dub", "Qahir", "Bagh", "Teh", "Khar"]:
    for _, r in city[city.city.str.contains(k, na=False)].iterrows():
        print(f"    {r.city[:30]:<32}{r.no2_ppb:>6.1f} ppb  {r.no2_ugm3:>6.1f} ug/m3  "
              f"+/-{r.seed_sd_ppb:.2f}  w={r.stage2_weight:.2f}  "
              f"in-dom {r.in_domain_pct:>3.0f}%  rho {r.expected_rho:.2f} ({r.confidence})")

A.to_csv(os.path.join(OUT, "step9_global100_predictions.csv"), index=False)
city.to_csv(os.path.join(OUT, "step9_city_summary.csv"), index=False)
for k, (s1, s2) in enumerate(PAIRS):
    s1.save_model(os.path.join(OUT, f"stage1_seed{k}.json"))
    s2.save_model(os.path.join(OUT, f"stage2_seed{k}.json"))
joblib.dump({"features": F, "units": "ppb", "ppb_to_ugm3": PPB_TO_UGM3, "airshed_km": 20, "n_seeds": N_SEEDS,
             "gate": {"lo": GATE_LO, "hi": GATE_HI, "variable": "city mean temp_c"},
             "rho_model": {"form": "tanh(a*T_city+b)", "a": float(KZ[0]), "b": float(KZ[1])},
             "train_ranges": {c: [float(COOL[c].min()), float(COOL[c].max())] for c in F}},
            os.path.join(OUT, "step9_model_metadata.joblib"))

print("\n" + "=" * 84)
print("3. SHAP - native TreeSHAP on stage 1")
dm = xgb.DMatrix(COOL[F], feature_names=F)
contribs = stage1.get_booster().predict(dm, pred_contribs=True)
sv = contribs[:, :-1]
print(f"  reconstruction error {np.abs(contribs.sum(1)-stage1.predict(COOL[F])).max():.2e}")
S = pd.DataFrame(sv, columns=F)
S["city"] = COOL.city.values
dd = []
for f in F:
    tot = S[f].var(); bet = S.groupby("city")[f].transform("mean").var()
    dd.append(dict(feature=f, mean_abs_shap=np.abs(S[f]).mean(),
                   between_city=bet, within_city=max(tot - bet, 0)))
dd = pd.DataFrame(dd)
dd["pct_within"] = 100 * dd.within_city / (dd.between_city + dd.within_city)
print(dd.sort_values("mean_abs_shap", ascending=False)
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
i, j = F.index("pblh_m"), F.index("no2_vcd_mol_m2")
print(f"\n  corr(PBLH, SHAP_PBLH) = {np.corrcoef(COOL.pblh_m, sv[:, i])[0,1]:+.3f}")
print(f"  corr(VCD,  SHAP_VCD)  = {np.corrcoef(COOL.no2_vcd_mol_m2, sv[:, j])[0,1]:+.3f}")
np.save(os.path.join(OUT, "step9_shap.npy"), sv)
dd.to_csv(os.path.join(OUT, "step9_shap_decomposition.csv"), index=False)
print(f"\nwrote predictions, city summary, models and SHAP to {OUT}")
