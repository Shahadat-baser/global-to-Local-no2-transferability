"""
STEP 7 - XGBOOST INVERSION  (CORRECTED)
========================================
    python step7_inversion.py

WHAT CHANGED AND WHY
  The first run included sin(month) and cos(month), added after the Step 6 EDA
  found strong seasonality. That was a mistake for an INVERSION model: the
  harmonics let the tree reproduce a city's seasonal climatology from the
  calendar instead of from the satellite column. cos(month) came out as the
  top SHAP feature - a physically incoherent result for a column-to-surface
  inversion.

  Removing them makes the model both better AND physically correct:

                              with harmonics   without
      P1 leave-one-city-out       0.526         0.537
      top SHAP feature          cos(month)     NO2 column
      value of the NO2 column     +0.087        +0.164
      corr(PBLH, SHAP_PBLH)       -0.315        -0.740

  Seasonality still enters the model - through temperature, PBLH and the
  column itself, which is where it physically belongs.

PROTOCOL (locked in Step 3, unchanged)
  P1  leave-one-city-out      PRIMARY - skill in a city with no monitors
  P2  leave-one-region-out    transfer between regions
  P3  temporal hold-out       gap-filling only; restricted to 2022-2025

TARGET
  Raw ppb. The log target was tested and is within noise of raw (0.525 vs
  0.526) with a larger negative bias, so raw is used. This was a
  pre-registered prediction from the EDA that did not hold - report it.

DATASET C (India) is NOT touched here.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (STEP1, STEP2, FINAL, CACHE, RAW, FIGDIR,
                   GEE_PROJECT, OPENAQ_API_KEY, require)
import os, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

FINAL = str(FINAL)
B = pd.read_csv(os.path.join(FINAL, "DATASET_B_train.csv"))
B["region"] = B.region.str.upper()

# ---- EDA-driven feature decisions -------------------------------------------
#   pop_density_per_km2 dropped : sign flips between regions
#   raw no2_sd -> no2_sd / VCD  : raw SD is 0.91-correlated with the column
#   NO seasonal harmonics       : see header
B["sd_ratio"] = B.no2_sd / B.no2_vcd_mol_m2
FEATS = ["no2_vcd_mol_m2", "pblh_m", "temp_c", "wind_u_ms", "wind_v_ms",
         "wind_speed_ms", "population", "built_up_km2", "sd_ratio",
         "cloud_fraction"]
TGT = "ground_no2_ppb"
y = B[TGT]
SEEDS = range(6)

print(f"DATASET B: {len(B):,} rows | {B.city.nunique()} cities")
print(B.groupby("region").agg(cities=("city", "nunique"),
                              rows=("city", "size")).to_string())


def mk(seed=42):
    return xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                            min_child_weight=5, tree_method="hist",
                            random_state=seed, n_jobs=4)


def show(n, yy, pp):
    print(f"  {n:<38} R2={r2_score(yy,pp):>6.3f}  "
          f"RMSE={np.sqrt(mean_squared_error(yy,pp)):>5.2f}  "
          f"MAE={mean_absolute_error(yy,pp):>5.2f}  "
          f"bias={np.mean(pp-yy):>+5.2f}")
    return r2_score(yy, pp)


def loco(feats=FEATS, seed=42):
    oof = np.zeros(len(B))
    for tr, te in GroupKFold(n_splits=B.city.nunique()).split(B[feats], y, B.city):
        oof[te] = mk(seed).fit(B[feats].iloc[tr], y.iloc[tr]).predict(B[feats].iloc[te])
    return oof


print("\n" + "=" * 80)
print("A. IS THE SATELLITE COLUMN ACTUALLY DOING THE WORK?")
print("   (the check that exposed the seasonal-harmonics error)")
full = show("full model", y, loco(FEATS))
nosat = show("WITHOUT the NO2 column", y, loco([f for f in FEATS if f != "no2_vcd_mol_m2"]))
print(f"  -> the satellite column is worth {full - nosat:+.3f} R2")
print("     If this were near zero, the model would not be an inversion.")

print("\n" + "=" * 80)
print("B. BASELINES")
show("persistence (city mean)", y, B.groupby("city")[TGT].transform("mean"))
show("monthly climatology", y, B.groupby("month")[TGT].transform("mean"))
show("linear on VCD", y, np.polyval(np.polyfit(B.no2_vcd_mol_m2, y, 1), B.no2_vcd_mol_m2))

print("\n" + "=" * 80)
print("C. P1 - LEAVE-ONE-CITY-OUT  (PRIMARY)")
oofs = np.column_stack([loco(FEATS, s) for s in SEEDS])
B["oof"] = oofs.mean(axis=1)
sc = np.array([r2_score(y, oofs[:, i]) for i in range(oofs.shape[1])])
show("XGBoost, city-blocked", y, B.oof)
print(f"  across {len(sc)} seeds: R2 = {sc.mean():.3f} +/- {sc.std():.3f}  "
      f"[{sc.min():.3f}, {sc.max():.3f}]")
for r, g in B.groupby("region"):
    show(f"  ...within {r}", g[TGT], g.oof)

print("\n" + "=" * 80)
print("D. P2 - LEAVE-ONE-REGION-OUT  (transfer)")
print("   With 4 regions, (ii) is the one that matters: can a model trained on")
print("   the other three predict a region it has never seen? That is Riyadh.")
regs = sorted(B.region.unique())
print("  (i) train on ONE region, test on the rest")
for a in regs:
    tr, te = B[B.region == a], B[B.region != a]
    if len(tr) < 100:
        print(f"  {a:<38} skipped - only {len(tr)} rows"); continue
    show(f"train {a} -> rest", te[TGT], mk().fit(tr[FEATS], tr[TGT]).predict(te[FEATS]))
print("  (ii) HOLD OUT one region, train on the others  <- the deployment case")
for a in regs:
    tr, te = B[B.region != a], B[B.region == a]
    if len(te) < 50:
        print(f"  {a:<38} skipped - only {len(te)} test rows"); continue
    show(f"hold out {a}", te[TGT], mk().fit(tr[FEATS], tr[TGT]).predict(te[FEATS]))

print("\n" + "=" * 80)
print("E. P3 - TEMPORAL HOLD-OUT (2022-2025 only; China has no earlier data)")
S = B[B.year >= 2022]
tr, te = S[S.year <= 2023], S[S.year >= 2024]
show("train 2022-23 -> test 2024-25", te[TGT], mk().fit(tr[FEATS], tr[TGT]).predict(te[FEATS]))

print("\n" + "=" * 80)
print("F. FEATURE BLOCKS (P1, cumulative)")
blocks = {
    "VCD only": ["no2_vcd_mol_m2"],
    "+ meteorology": ["no2_vcd_mol_m2", "pblh_m", "temp_c", "wind_u_ms",
                      "wind_v_ms", "wind_speed_ms"],
    "+ urban": ["no2_vcd_mol_m2", "pblh_m", "temp_c", "wind_u_ms", "wind_v_ms",
                "wind_speed_ms", "population", "built_up_km2"],
    "+ QA (full)": FEATS,
}
for n, f in blocks.items():
    show(n, y, loco(f))

print("\n" + "=" * 80)
print("G. SKILL DECOMPOSITION")
tot = y.var(); bet = B.groupby("city")[TGT].transform("mean").var()
print(f"  between-city variance {100*bet/tot:.1f}% | within-city {100*(tot-bet)/tot:.1f}%")
ya = y - B.groupby("city")[TGT].transform("mean")
pa = B.oof - B.groupby("city").oof.transform("mean")
print(f"  within-city anomaly R2 : {r2_score(ya, pa):.3f}   <- tracks the cycle")
lev = B.groupby("city").agg(o=(TGT, "mean"), p=("oof", "mean"))
print(f"  between-city level R2  : {r2_score(lev.o, lev.p):.3f}   <- cannot set the baseline")

print("\n" + "=" * 80)
print("H. PER-CITY SKILL (mean +/- sd across seeds)")
rows = []
for c in sorted(B.city.unique()):
    m = (B.city == c).values
    rs = np.array([r2_score(y[m], oofs[m, i]) for i in range(oofs.shape[1])])
    lvl = B.oof[m].mean() - y[m].mean()
    shp = np.corrcoef(B.oof[m], y[m])[0, 1] ** 2
    rows.append((c, B.region[m].iloc[0], m.sum(), rs.mean(), rs.std(),
                 np.sqrt(mean_squared_error(y[m], B.oof[m])), lvl, shp, y[m].mean()))
pc = pd.DataFrame(rows, columns=["city", "region", "n", "R2", "R2_sd", "RMSE",
                                 "level_bias_ppb", "shape_r2", "mean_ppb"])
pc = pc.sort_values("R2", ascending=False)
print(pc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\n  positive skill: {(pc.R2>0).sum()}/{len(pc)} cities")
bad, good = pc[pc.R2 < 0], pc[pc.R2 >= 0]
if len(bad):
    print(f"  failing cities: mean |level bias| {bad.level_bias_ppb.abs().mean():.2f} ppb, "
          f"shape r2 {bad.shape_r2.mean():.2f}")
    print(f"  passing cities: mean |level bias| {good.level_bias_ppb.abs().mean():.2f} ppb, "
          f"shape r2 {good.shape_r2.mean():.2f}")
    print("  -> failures are LEVEL errors, not shape errors.")

print("\n" + "=" * 80)
print("I. FINAL MODEL + SHAP")
m = mk().fit(B[FEATS], y)
joblib.dump({"model": m, "features": FEATS, "units": "ppb", "airshed_km": 20,
             "train_ranges": {c: [float(B[c].min()), float(B[c].max())] for c in FEATS}},
            os.path.join(FINAL, "step7_model.joblib"))
# TreeSHAP computed by XGBoost itself (pred_contribs=True), NOT the shap
# library. XGBoost >= 3.0 writes base_score as a JSON array - '[1.1115915E1]' -
# which older shap builds cannot parse:
#     ValueError: could not convert string to float: '[1.1115915E1]'
# The native path returns identical values, has no extra dependency, and cannot
# drift out of step with the XGBoost version.
dm = xgb.DMatrix(B[FEATS], feature_names=FEATS)
contribs = m.get_booster().predict(dm, pred_contribs=True)
sv = contribs[:, :-1]                      # last column is the base value

# sanity: contributions must sum to the prediction
recon = np.abs(contribs.sum(1) - m.predict(B[FEATS])).max()
print(f"  TreeSHAP reconstruction error: {recon:.2e}  (should be ~0)")

imp = pd.DataFrame({"feature": FEATS, "mean_abs_shap": np.abs(sv).mean(0)}
                   ).sort_values("mean_abs_shap", ascending=False)
print(imp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
i, j = FEATS.index("pblh_m"), FEATS.index("no2_vcd_mol_m2")
print(f"\n  corr(PBLH, SHAP_PBLH) = {np.corrcoef(B.pblh_m, sv[:, i])[0,1]:+.3f}"
      "   (negative = lid effect, physically consistent)")
print(f"  corr(VCD,  SHAP_VCD)  = {np.corrcoef(B.no2_vcd_mol_m2, sv[:, j])[0,1]:+.3f}"
      "   (positive = more column, more surface)")
if imp.feature.iloc[0] != "no2_vcd_mol_m2":
    print(f"\n  WARNING: top feature is '{imp.feature.iloc[0]}', not the NO2 column.")
    print("  For a column-to-surface INVERSION that needs explaining - check")
    print("  section A above: how much is the satellite column actually worth?")
np.save(os.path.join(FINAL, "shap_values.npy"), sv)
imp.to_csv(os.path.join(FINAL, "step7_shap.csv"), index=False)

pc.to_csv(os.path.join(FINAL, "step7_per_city.csv"), index=False)
B.to_csv(os.path.join(FINAL, "step7_oof.csv"), index=False)
json.dump({"feats": FEATS, "p1_mean": float(sc.mean()), "p1_sd": float(sc.std()),
           "n_cities": int(B.city.nunique()), "regions": regs},
          open(os.path.join(FINAL, "step7_cfg.json"), "w"), indent=2)
print(f"\nwrote model, per-city table, OOF predictions and config to {FINAL}")
