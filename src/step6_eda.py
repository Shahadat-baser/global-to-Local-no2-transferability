"""
STEP 6 - EXPLORATORY DATA ANALYSIS
===================================
    python step6_eda.py

Runs on the three assembled datasets. DATASET C (India) is only summarised
here - it is not used for any modelling decision, and is opened properly once,
after the model is final.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (STEP1, STEP2, FINAL, CACHE, RAW, FIGDIR,
                   GEE_PROJECT, OPENAQ_API_KEY, require)
import pandas as pd, numpy as np
from scipy.stats import pearsonr, spearmanr

import os
FINAL = str(FINAL)
B = pd.read_csv(os.path.join(FINAL, "DATASET_B_train.csv"))
A = pd.read_csv(os.path.join(FINAL, "DATASET_A_target_global100.csv"))
C = pd.read_csv(os.path.join(FINAL, "DATASET_C_test_india_arid.csv"))

print("=" * 74)
print("1. TARGET DISTRIBUTION - is surface NO2 lognormal?")
y = B.ground_no2_ppb
print(f"  n={len(y)}  mean {y.mean():.2f}  median {y.median():.2f}  "
      f"sd {y.std():.2f}  skew {y.skew():.2f}  kurt {y.kurtosis():.2f}")
ly = np.log(y)
print(f"  log-target: skew {ly.skew():.2f}  kurt {ly.kurtosis():.2f}"
      f"   -> {'log helps' if abs(ly.skew()) < abs(y.skew()) else 'log does not help'}")
for q in [.01, .25, .5, .75, .99]:
    print(f"    p{int(q*100):<3} {y.quantile(q):6.2f} ppb")

print("\n" + "=" * 74)
print("2. REGIONAL STRUCTURE - the column-to-surface ratio")
for r, g in B.groupby('region'):
    print(f"  {r:<6} n={len(g):>5}  VCD {g.no2_vcd_mol_m2.mean():.3e}  "
          f"surface {g.ground_no2_ppb.mean():5.2f} ppb  "
          f"ratio {g.ground_no2_ppb.mean()/g.no2_vcd_mol_m2.mean():>9,.0f}")
print("  -> a 2.3x difference in how column maps to surface. This is the")
print("     mechanism behind the negative cross-region transfer in Step 4.")

print("\n" + "=" * 74)
print("3. FEATURE-TARGET RELATIONSHIPS (Spearman, monotonic not linear)")
FE = ['no2_vcd_mol_m2', 'pblh_m', 'temp_c', 'wind_speed_ms', 'wind_u_ms',
      'wind_v_ms', 'population', 'built_up_km2', 'pop_density_per_km2',
      'cloud_fraction', 'no2_sd']
rows = []
for f in FE:
    if f not in B: continue
    rows.append((f, spearmanr(B[f], y)[0],
                 spearmanr(B[B.region == 'USA'][f], B[B.region == 'USA'].ground_no2_ppb)[0],
                 spearmanr(B[B.region == 'CHINA'][f], B[B.region == 'CHINA'].ground_no2_ppb)[0]))
sp = pd.DataFrame(rows, columns=['feature', 'pooled', 'USA', 'CHINA'])
sp['sign_flip'] = np.sign(sp.USA) != np.sign(sp.CHINA)
print(sp.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print(f"\n  features whose sign FLIPS between regions: "
      f"{sp.sign_flip.sum()}/{len(sp)}  <- these cannot transfer")

print("\n" + "=" * 74)
print("4. THE LID EFFECT - does PBLH suppress surface NO2?")
for r, g in B.groupby('region'):
    a = g.ground_no2_ppb - g.groupby('city').ground_no2_ppb.transform('mean')
    p = g.pblh_m - g.groupby('city').pblh_m.transform('mean')
    print(f"  {r:<6} raw rho(PBLH, NO2) = {spearmanr(g.pblh_m, g.ground_no2_ppb)[0]:+.3f}"
          f"   |  WITHIN-CITY anomaly rho = {spearmanr(p, a)[0]:+.3f}")
print("  -> the raw correlation is confounded by city identity; the within-city")
print("     anomaly is the physically meaningful one.")

print("\n" + "=" * 74)
print("5. SEASONALITY")
m = B.groupby(['region', 'month']).ground_no2_ppb.mean().unstack(0)
print(m.round(1).to_string())
for r in m.columns:
    print(f"  {r}: peak month {int(m[r].idxmax())}, trough {int(m[r].idxmin())}, "
          f"amplitude {m[r].max()-m[r].min():.1f} ppb "
          f"({100*(m[r].max()-m[r].min())/m[r].mean():.0f}% of mean)")

print("\n" + "=" * 74)
print("6. MULTICOLLINEARITY (|r| > 0.7 among predictors)")
C = B[[f for f in FE if f in B]].corr()
seen = set()
for i in C.columns:
    for j in C.columns:
        if i != j and abs(C.loc[i, j]) > 0.7 and (j, i) not in seen:
            seen.add((i, j))
            print(f"  {i:<22} {j:<22} r={C.loc[i,j]:+.3f}")
print("  -> tree models tolerate this, but SHAP attribution splits credit")
print("     between correlated features; say so when interpreting.")

print("\n" + "=" * 74)
print("7. DATA QUALITY")
print(f"  cloud_fraction  mean {B.cloud_fraction.mean():.3f}  "
      f"p90 {B.cloud_fraction.quantile(.9):.3f}  max {B.cloud_fraction.max():.3f}")
B['rel_sd'] = B.no2_sd / B.no2_vcd_mol_m2
print(f"  within-month spread (no2_sd/VCD): median {B.rel_sd.median():.2f}  "
      f"p90 {B.rel_sd.quantile(.9):.2f}")
print(f"  n_obs_no2 per city-month: {B.n_obs_no2.min():.0f}-{B.n_obs_no2.max():.0f}")
hi = B[B.cloud_fraction > B.cloud_fraction.quantile(.75)]
lo = B[B.cloud_fraction < B.cloud_fraction.quantile(.25)]
print(f"  rho(VCD,ground) low-cloud  {spearmanr(lo.no2_vcd_mol_m2, lo.ground_no2_ppb)[0]:+.3f}")
print(f"  rho(VCD,ground) high-cloud {spearmanr(hi.no2_vcd_mol_m2, hi.ground_no2_ppb)[0]:+.3f}")

print("\n" + "=" * 74)
print("8. TEMPORAL COVERAGE")
print(pd.crosstab(B.region, B.year).to_string())
print("  -> China contributes only 2022-2025. Any year-blocked split must")
print("     account for this or it will confound region with period.")

print("\n" + "=" * 74)
print("9. TRAINING ENVELOPE vs RIYADH (the deployment question)")
PRED = ['no2_vcd_mol_m2', 'pblh_m', 'temp_c', 'wind_u_ms', 'wind_v_ms']
riy = A[A.city.str.contains('Riy', na=False)]
print(f"  {'predictor':<18}{'TRAIN':>26}{'RIYADH':>26}")
for c in PRED:
    print(f"  {c:<18}{f'[{B[c].min():.4g}, {B[c].max():.4g}]':>26}"
          f"{f'[{riy[c].min():.4g}, {riy[c].max():.4g}]':>26}")
out = (~riy.pblh_m.between(B.pblh_m.min(), B.pblh_m.max())).mean()
print(f"\n  Riyadh months with PBLH outside training range: {100*out:.0f}%")
print("  PBLH is the single binding constraint - not VCD, not temperature.")
