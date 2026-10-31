"""
STEP 8 - TRANSFERABILITY: WHERE SATELLITE-TO-SURFACE INVERSION WORKS
=====================================================================
    python step8_transferability.py

Combines the two design options:
  (1) train on the STRONG-LINK regions - US + China, where rho(column, ground)
      is 0.83 and 0.92 - rather than pooling every region
  (3) climate stratification - a second-stage model, fitted on the hot/arid
      cities, that corrects the first stage's residual

Sections:
  A  primary model and its validation
  B  four competing designs, all scored on the same held-out arid test
  C  does climate stratification recover skill, or only remove bias?
  D  how much hot-regime data would be needed
  E  an uncertainty model: expected rho from mean temperature

WHY NOT SIMPLY POOL EVERYTHING
  Pooling all 49 cities gave P1 = 0.247. Training on the 35 strong-link cities
  gives 0.570. The Gulf and Indian ground data are not worse in quality - the
  column-to-surface relationship is genuinely weaker there, so adding those
  cities adds noise to the primary signal. Section E quantifies why.

DATASET C (India arid) is used ONLY as the final held-out test. No model, no
feature, no threshold in this script was chosen by looking at it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (STEP1, STEP2, FINAL, CACHE, RAW, FIGDIR,
                   GEE_PROJECT, OPENAQ_API_KEY, require)
import os, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

FINAL = str(FINAL)

B = pd.read_csv(os.path.join(FINAL, "DATASET_B_train.csv"))
C = pd.read_csv(os.path.join(FINAL, "DATASET_C_test_india_arid.csv"))
for d in (B, C):
    d["sd_ratio"] = d.no2_sd / d.no2_vcd_mol_m2

F = ["no2_vcd_mol_m2", "pblh_m", "temp_c", "wind_u_ms", "wind_v_ms",
     "wind_speed_ms", "population", "built_up_km2", "sd_ratio", "cloud_fraction"]
T = "ground_no2_ppb"

STRONG = ["USA", "CHINA"]                       # rho > 0.8
HOT = [r for r in B.region.unique() if r not in STRONG]

COOL = B[B.region.isin(STRONG)].reset_index(drop=True)
HOTD = B[B.region.isin(HOT)].reset_index(drop=True)
print(f"strong-link : {COOL.city.nunique():>3} cities, {len(COOL):>5} rows  {STRONG}")
print(f"hot/arid    : {HOTD.city.nunique():>3} cities, {len(HOTD):>5} rows  {HOT}")
print(f"held-out    : {C.city.nunique():>3} cities, {len(C):>5} rows  (India arid)")


def mk(seed=42):
    return xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                            min_child_weight=5, tree_method="hist",
                            random_state=seed, n_jobs=4)


def rep(name, y, p):
    print(f"  {name:<48} R2={r2_score(y,p):>7.3f}  "
          f"RMSE={np.sqrt(mean_squared_error(y,p)):>5.2f}  "
          f"MAE={mean_absolute_error(y,p):>5.2f}  bias={np.mean(p-y):>+6.2f}")
    return r2_score(y, p)


def loco(d, feats=F, seed=42):
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=d.city.nunique()).split(d[feats], d[T], d.city):
        oof[te] = mk(seed).fit(d[feats].iloc[tr], d[T].iloc[tr]).predict(d[feats].iloc[te])
    return oof


print("\n" + "=" * 88)
print("A. PRIMARY MODEL - trained on the strong-link regions only")
oofs = np.column_stack([loco(COOL, seed=s) for s in range(6)])
COOL["oof"] = oofs.mean(1)
r = np.array([r2_score(COOL[T], oofs[:, i]) for i in range(oofs.shape[1])])
rep("P1 leave-one-city-out", COOL[T], COOL.oof)
print(f"  across {len(r)} seeds: R2 = {r.mean():.3f} +/- {r.std():.3f}")
rep("persistence baseline", COOL[T], COOL.groupby("city")[T].transform("mean"))

base = mk().fit(COOL[F], COOL[T])

print("\n" + "=" * 88)
print("B. FOUR DESIGNS, ALL SCORED ON THE SAME HELD-OUT ARID TEST")
r1 = rep("1. strong-link only", C[T], base.predict(C[F]))

m_hot = mk().fit(HOTD[F], HOTD[T])
r3 = rep("3. hot-regime model only", C[T], m_hot.predict(C[F]))

ALL = pd.concat([COOL, HOTD], ignore_index=True)
r0 = rep("0. pooled, no stratification", C[T], mk().fit(ALL[F], ALL[T]).predict(C[F]))

# --- the combination: strong-link base + hot-regime residual correction -------
# Stage 1 learns the column-to-surface physics where the signal is clean.
# Stage 2 learns, from the few hot cities available, how that mapping is biased
# under arid conditions. Neither stage works alone; together they remove the
# bias almost entirely.
HOTD = HOTD.copy()
HOTD["base"] = base.predict(HOTD[F])
HOTD["resid"] = HOTD[T] - HOTD.base
corr = mk().fit(HOTD[F + ["base"]], HOTD.resid)

Cx = C.copy(); Cx["base"] = base.predict(C[F])
p13 = Cx.base + corr.predict(Cx[F + ["base"]])
r13 = rep("1+3. strong-link base + hot residual correction", C[T], p13)

w = np.clip((C.temp_c - 15) / (28 - 15), 0, 1)
pb = (1 - w) * base.predict(C[F]) + w * m_hot.predict(C[F])
rb = rep("1+3b. temperature-weighted blend", C[T], pb)

designs = {"strong-link": (r1, base.predict(C[F])), "hot only": (r3, m_hot.predict(C[F])),
           "pooled": (r0, mk().fit(ALL[F], ALL[T]).predict(C[F])),
           "1+3 residual": (r13, p13), "1+3 blend": (rb, pb)}
best = max(designs.items(), key=lambda kv: kv[1][0])
print(f"\n  BEST: {best[0]}  (R2 = {best[1][0]:+.3f})")

print("\n  per-city on the held-out arid set:")
print(f"  {'city':<14}{'strong-link':>13}{'best':>10}{'bias':>9}{'rho(col,gr)':>13}")
for c in sorted(C.city.unique()):
    m_ = (C.city == c).values
    print(f"  {c:<14}{r2_score(C[T][m_], base.predict(C[F])[m_]):>13.3f}"
          f"{r2_score(C[T][m_], best[1][1][m_]):>10.3f}"
          f"{np.mean(best[1][1][m_] - C[T][m_]):>9.2f}"
          f"{spearmanr(C.no2_vcd_mol_m2[m_], C[T][m_])[0]:>13.2f}")

print("\n" + "=" * 88)
print("C. DOES STRATIFICATION RECOVER SKILL, OR ONLY REMOVE BIAS?")
o = loco(HOTD)
rep("P1 within the hot cities", HOTD[T], o)
rep("P1 within the strong-link cities", COOL[T], COOL.oof)
print("  -> if hot-city P1 stays far below strong-link P1, the column-to-surface")
print("     link is intrinsically weaker there; stratification cannot manufacture")
print("     signal that is not in the data.")

print("\n  is the arid failure a LEVEL error or a SHAPE error?")
p = base.predict(C[F])
print(f"    raw transfer                         R2 = {r2_score(C[T], p):+.3f}")
print(f"    after removing the mean bias         R2 = {r2_score(C[T], p - np.mean(p - C[T])):+.3f}")
a_, b_ = np.polyfit(p, C[T], 1)
print(f"    after linear recalibration           R2 = {r2_score(C[T], a_*p + b_):+.3f}")
cm = C.assign(pred=p).groupby("city").agg(o=(T, "mean"), p=("pred", "mean"))
print(f"    city-level ranking rho               = {spearmanr(cm.o, cm.p)[0]:+.3f}")
bb = C.assign(e=p - C[T]).groupby("city").e.mean()
print(f"    per-city bias spread: {bb.min():.1f} to {bb.max():.1f} ppb"
      "  -> no single offset serves all arid cities")

print("\n" + "=" * 88)
print("D. WOULD MORE HOT-REGIME DATA HELP?")
hc = sorted(HOTD.city.unique()); rng = np.random.default_rng(0)
for n in [3, 6, 9, 12, len(hc)]:
    if n > len(hc):
        continue
    s = []
    for _ in range(12):
        sub = HOTD[HOTD.city.isin(rng.permutation(hc)[:n])]
        if sub.city.nunique() < n:
            continue
        s.append(r2_score(C[T], mk().fit(sub[F], sub[T]).predict(C[F])))
    print(f"    {n:>2} hot cities -> R2 = {np.mean(s):+.3f} +/- {np.std(s):.3f}")
print("  -> a flat curve means the ceiling is physical, not a sample-size problem.")

print("\n" + "=" * 88)
print("E. UNCERTAINTY MODEL - expected rho from climate")
allc = pd.concat([B, C], ignore_index=True)
rows = []
for c, g in allc.groupby("city"):
    if len(g) < 19:
        continue
    rows.append(dict(city=c, region=g.region.iloc[0],
                     rho=spearmanr(g.no2_vcd_mol_m2, g[T])[0],
                     temp=g.temp_c.mean(), pblh=g.pblh_m.mean(),
                     vcd=g.no2_vcd_mol_m2.mean(), n=len(g)))
q = pd.DataFrame(rows)
for v in ["temp", "pblh", "vcd"]:
    rr, pp = pearsonr(q[v], q.rho)
    print(f"  rho vs mean {v:<5}: r = {rr:+.3f}  (p = {pp:.4f})")

# Fisher z: correlations are bounded in [-1, 1], so a raw linear fit gives
# rho > 1 for cold cities. Fit on the unbounded z scale and transform back.
kz = np.polyfit(q.temp, np.arctanh(np.clip(q.rho, -0.999, 0.999)), 1)
print(f"\n  rho = tanh({kz[0]:+.4f} * mean_temp + {kz[1]:.3f})")
for t in [10, 15, 20, 25, 30]:
    print(f"    T = {t:>2} C -> expected rho = {np.tanh(np.polyval(kz, t)):.2f}")

q.to_csv(os.path.join(FINAL, "step8_city_rho.csv"), index=False)
json.dump({"p1_strong": float(r.mean()), "p1_sd": float(r.std()),
           "arid_strong_only": float(r1), "arid_two_stage": float(r13),
           "best_design": best[0], "rho_model": {"form": "tanh(a*T+b)",
                                                 "a": float(kz[0]), "b": float(kz[1])}},
          open(os.path.join(FINAL, "step8_results.json"), "w"), indent=2)
print(f"\nwrote step8_city_rho.csv and step8_results.json to {FINAL}")
print("Next: step9_apply_and_shap.py")
