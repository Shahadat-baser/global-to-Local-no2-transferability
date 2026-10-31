"""
STEP 2e - MASTER REBUILD (all cities, one vintage, multi-radius)
================================================================
Rebuilds EVERY city - the 100 global target cities and the 40 ground-truth
training cities - through one code path, and extracts TWO airshed radii in the
same pass so the radius can be chosen by validation instead of by convention.

    python step2e_master_rebuild.py            # edit WHICH first

WHICH = "GLOBAL100" | "USA" | "CHINA"

WHY TWO RADII
  Ground stations sit at a median of 15 km from the city centroid (measured
  from the CNEMC network), but the 90th percentile is 39 km - the tail is long
  and varies hugely by city (Hefei 13 km, Chongqing 192 km). No single radius
  is obviously right, so extract 20 km and 30 km together and let
  leave-one-city-out decide. The marginal cost is one extra reduceRegions.

WHAT IS DROPPED
  SO2 - EPA ground SO2 sits at/below the AQS detection floor (89% of readings
  under 1 ppb, 2.4% negative, column-ground r = +0.12). The SO2 column is still
  extracted as a covariate, but SO2 inversion is out of scope.

WHAT IS KEPT (unchanged objectives)
  1. satellite-to-surface inversion for surface NO2
  2. deep-learning forecasting / "Virtual Station" reconstruction
  3. transferability to data-scarce arid cities (Riyadh, Jeddah, Dammam)

PBLH WARNING - READ THIS
  MERRA-2 publishes TWO boundary-layer heights:
      PBLH    eddy-diffusion (Kh) definition   <- used here
      TCZPBL  bulk Richardson number, runs much deeper
  Your old extraction gave 131-1465 m; this pipeline gives 210-2025 m over the
  same cities and months (35% median difference). That is a definitional gap,
  not noise. PBLH is the paper's central mechanism, so ONE definition must be
  used everywhere and named explicitly in Methods. Set PBLH_BAND below.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (STEP1, STEP2, FINAL, CACHE, RAW, FIGDIR,
                   GEE_PROJECT, OPENAQ_API_KEY, require)
import ee, os, sys, time
import pandas as pd

# ------------------------------------------------------------------ settings -
PROJECT = require("GEE_PROJECT")
WHICH     = "GLOBAL100"                               # <<< EDIT per run
S1 = str(STEP1)
OUTDIR = str(STEP2)

PBLH_BAND = "PBLH"          # "PBLH" (eddy diffusion) or "TCZPBL" (Richardson)
RADII_M   = [20000, 30000]  # extracted together; choose later by validation
START, END = "2019-01-01", "2026-01-01"
YEARS      = range(2019, 2026)
S5P_SCALE, MERRA_SCALE = 1113.2, 50000

SRC = {
    "GLOBAL100": (os.path.join(S1, "global100_cities.csv"), "GLOBAL100"),
    "USA":       (os.path.join(S1, "usa_cities_for_extraction.csv"), "USA"),
    "CHINA":     (os.path.join(S1, "china_cities_for_extraction.csv"), "CHINA"),
}[WHICH]
CITY_CSV, TAG = SRC

os.makedirs(OUTDIR, exist_ok=True)
ee.Initialize(project=PROJECT)

cities = pd.read_csv(CITY_CSV)
NAME_COL = "city_en" if "city_en" in cities.columns else "City_Name"
LAT_COL = next(c for c in cities.columns if "at" in c.lower() and cities[c].dtype.kind == "f")
LON_COL = next(c for c in cities.columns if "on" in c.lower() and cities[c].dtype.kind == "f")

pts = [(str(r[NAME_COL]), float(r[LAT_COL]), float(r[LON_COL]))
       for _, r in cities.iterrows()]
AOI = {R: ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lo, la]).buffer(R), {"city": n})
        for n, la, lo in pts]) for R in RADII_M}

months = pd.date_range(START, END, freq="MS")[:-1]
print(f"{TAG}: {len(pts)} cities | radii {[r//1000 for r in RADII_M]} km | "
      f"{len(months)} months | PBLH band = {PBLH_BAND}\n")


# ------------------------------------------------- stage 1: satellite + met --
def month_stats(dt, R):
    a = ee.Date(dt.strftime("%Y-%m-%d")); b = a.advance(1, "month")
    no2c = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
            .select(["tropospheric_NO2_column_number_density", "cloud_fraction"])
            .filterDate(a, b))
    so2c = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_SO2")
            .select("SO2_column_number_density").filterDate(a, b))
    flx = (ee.ImageCollection("NASA/GSFC/MERRA/flx/2")
           .select(PBLH_BAND).filterDate(a, b))
    slv = (ee.ImageCollection("NASA/GSFC/MERRA/slv/2")
           .select(["T2M", "U10M", "V10M"]).filterDate(a, b))

    sat = (no2c.mean().rename(["no2_vcd_mol_m2", "cloud_fraction"])
           .addBands(no2c.select("tropospheric_NO2_column_number_density")
                     .reduce(ee.Reducer.stdDev()).rename("no2_sd"))
           .addBands(so2c.mean().rename("so2_vcd_mol_m2")))
    met = flx.mean().rename("pblh_m").addBands(
        slv.mean().rename(["t2m_k", "wind_u_ms", "wind_v_ms"]))

    s = (sat.reduceRegions(collection=AOI[R], reducer=ee.Reducer.mean(),
                           scale=S5P_SCALE, tileScale=4)
         .map(lambda f: f.set("year", a.get("year"), "month", a.get("month"),
                              "n_obs_no2", no2c.size(), "airshed_km", R // 1000)))
    return (met.reduceRegions(collection=s, reducer=ee.Reducer.mean(),
                              scale=MERRA_SCALE, tileScale=4)
            .select([".*"], None, False))


print("[1/3] satellite + meteorology")
rows, failed = [], []
for R in RADII_M:
    print(f"  --- {R//1000} km airshed ---")
    for i, m in enumerate(months, 1):
        for attempt in range(3):
            try:
                rows += [f["properties"]
                         for f in month_stats(m, R).getInfo()["features"]]
                break
            except Exception as e:
                if attempt == 2:
                    failed.append((R // 1000, m.strftime("%Y-%m")))
                    print(f"    FAILED {m:%Y-%m}: {e}")
                else:
                    time.sleep(5 * (attempt + 1))
        if i % 12 == 0 or i == len(months):
            print(f"    {i:>3}/{len(months)}  rows={len(rows):,}", flush=True)

sat = pd.DataFrame(rows)
sat.to_csv(os.path.join(OUTDIR, f"DUMI_v3_{TAG}_satmet.csv"), index=False)


# ---------------------------------------- stage 2: GHSL urban (nodata-masked) -
def epoch_totals(epoch, R):
    built = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{epoch}").select("built_surface")
    pop = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{epoch}").select("population_count")
    built = built.updateMask(built.gte(0))
    pop = pop.updateMask(pop.gte(0))
    # unmask(0) BEFORE the mean, or water pixels are excluded and the
    # fraction comes back as 1 (or None) everywhere - the land_frac bug.
    land = pop.mask().unmask(0).rename("land_fraction")

    sums = (built.rename("built_up_m2").addBands(pop.rename("population"))
            .reduceRegions(collection=AOI[R], reducer=ee.Reducer.sum(),
                           scale=100, tileScale=8))
    out = land.reduceRegions(collection=sums, reducer=ee.Reducer.mean(),
                             scale=100, tileScale=8)
    return {f["properties"]["city"]: f["properties"]
            for f in out.getInfo()["features"]}


print("\n[2/3] GHSL urban features")
urows = []
for R in RADII_M:
    E = {ep: epoch_totals(ep, R) for ep in (2020, 2025)}
    neg = sum(1 for v in E[2020].values() if v.get("population", 0) < 0)
    print(f"  {R//1000} km: {len(E[2020])} cities | negatives: {neg}")
    for city in E[2020]:
        a, b = E[2020][city], E[2025][city]
        for y in YEARS:
            w = (y - 2020) / 5.0
            urows.append(dict(
                city=city, year=y, airshed_km=R // 1000,
                built_up_m2=a["built_up_m2"] + w * (b["built_up_m2"] - a["built_up_m2"]),
                population=a["population"] + w * (b["population"] - a["population"]),
                land_fraction=a.get("land_fraction"),
                ghsl_interpolated=(y not in (2020, 2025)),
                ghsl_projected=(y > 2025)))
urb = pd.DataFrame(urows)
urb["built_up_km2"] = urb.built_up_m2 / 1e6
urb["pop_density_per_km2"] = urb.population / urb.built_up_km2.replace(0, pd.NA)
urb.to_csv(os.path.join(OUTDIR, f"urban_v3_{TAG}.csv"), index=False)


# ------------------------------------------------- stage 3: merge + validate --
print("\n[3/3] merge and validate")
df = sat.merge(urb, on=["city", "year", "airshed_km"], how="left")
df["temp_c"] = df["t2m_k"] - 273.15
df["wind_speed_ms"] = (df.wind_u_ms ** 2 + df.wind_v_ms ** 2) ** 0.5
df["pblh_band"] = PBLH_BAND
df["s5p_qa_threshold"] = 75
df["extraction_date"] = pd.Timestamp.utcnow().date().isoformat()
df["pipeline_version"] = "v3"

for c, (lo, hi) in {"temp_c": (-50, 60), "pblh_m": (20, 6000),
                    "no2_vcd_mol_m2": (-1e-3, 5e-3)}.items():
    mn, mx = df[c].min(), df[c].max()
    assert lo <= mn and mx <= hi, f"UNIT VIOLATION {c}: [{mn}, {mx}]"
    print(f"  ok  {c}: [{mn:.4g}, {mx:.4g}]")
assert (df.population > 0).all(), "negative population - do not proceed"
assert df.duplicated(["city", "year", "month", "airshed_km"]).sum() == 0

out = os.path.join(OUTDIR, f"DUMI_v3_{TAG}_complete.csv")
df.to_csv(out, index=False)
print(f"\nwrote {out}")
print(f"  {len(df):,} rows | {df.city.nunique()} cities | "
      f"radii {sorted(df.airshed_km.unique())}")
print(f"  land_fraction populated: {df.land_fraction.notna().mean()*100:.0f}%")

print("\n20 km vs 30 km - how different is the satellite signal?")
piv = df.pivot_table(index=["city", "year", "month"], columns="airshed_km",
                     values="no2_vcd_mol_m2").dropna()
if piv.shape[1] == 2:
    a, b = piv.columns
    rel = 100 * (piv[a] - piv[b]).abs() / piv[b]
    print(f"  median |20km - 30km| / 30km = {rel.median():.1f}%")
    print(f"  correlation = {piv.corr().iloc[0,1]:.4f}")
    print("  -> if these are nearly identical, radius barely matters and 30 km")
    print("     stands; if they differ, pick by leave-one-city-out in Step 4.")
if failed:
    print(f"\n  re-run: {failed}")
