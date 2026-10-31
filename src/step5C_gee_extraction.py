"""
STEP 5C - GEE EXTRACTION for Gulf + India (training and held-out test)
======================================================================
    python step5C_gee_extraction.py

Runs ALL THREE regions in one go - no editing between runs. Set PROJECT below,
then execute once. To do just one region, edit REGIONS at the top.

  GULF         5 training cities  - Abu Dhabi, Al Ain, Ruwais, Bida Zayed, Hail
                                    hot desert; Hail reaches PBLH 3144 m, the
                                    only training city near Riyadh's 3309 m

  INDIA_TRAIN  10 training cities - Delhi, Kanpur, Lucknow, Patna, Varanasi,
                                    Pune, Hyderabad, Bengaluru, Kolkata, Bhopal
                                    Deccan + Indo-Gangetic. These carry the HIGH
                                    PBLH the US and China blocks lack
                                    (Pune 2727 m, Kanpur 2647 m vs a 2034 m
                                    ceiling from the US cities)

  INDIA_TEST   5 held-out cities  - Jaipur, Jodhpur, Kota, Ahmedabad,
                                    Gandhinagar. Arid north-west India, roughly
                                    1000 km from the training cities and in a
                                    different Koppen zone. NEVER used for
                                    fitting, tuning or feature selection.

Identical code path, QA rules, airshed radii, PBLH band and unit handling as the
GLOBAL100 / USA / CHINA runs, so all cities sit on pipeline v3.

INPUTS  (all in step1_outputs\\)
    gulf_cities_for_extraction.csv
    india_train_cities_for_extraction.csv
    india_test_cities_for_extraction.csv

OUTPUTS (in step2_outputs\\)
    DUMI_v3_GULF_complete.csv
    DUMI_v3_INDIA_TRAIN_complete.csv
    DUMI_v3_INDIA_TEST_complete.csv
    (plus *_satmet.csv and urban_v3_*.csv intermediates)

RUNTIME  roughly 15 min for 5 cities, 30 min for 10 - about an hour in total.
Safe to re-run: a region whose _complete.csv already exists is skipped unless
FORCE = True.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (STEP1, STEP2, FINAL, CACHE, RAW, FIGDIR,
                   GEE_PROJECT, OPENAQ_API_KEY, require)
import ee, os, time
import pandas as pd

# ------------------------------------------------------------------ settings -
PROJECT = require("GEE_PROJECT")
S1 = str(STEP1)
OUTDIR = str(STEP2)
FORCE   = False          # True = re-extract even if the output already exists

REGIONS = ["GULF", "INDIA_TRAIN", "INDIA_TEST"]       # trim this to do one

CITY_FILES = {
    "GULF":        "gulf_cities_for_extraction.csv",
    "INDIA_TRAIN": "india_train_cities_for_extraction.csv",
    "INDIA_TEST":  "india_test_cities_for_extraction.csv",
}
REGION_TAG = {"GULF": "GULF", "INDIA_TRAIN": "INDIA_TRAIN", "INDIA_TEST": "INDIA_ARID"}
ROLE = {"GULF": "train", "INDIA_TRAIN": "train", "INDIA_TEST": "test"}

PBLH_BAND  = "PBLH"          # MUST match every other run
RADII_M    = [20000, 30000]  # 20 km selected in Step 4; 30 km kept for the record
START, END = "2019-01-01", "2026-01-01"
YEARS      = range(2019, 2026)
S5P_SCALE, MERRA_SCALE = 1113.2, 50000

os.makedirs(OUTDIR, exist_ok=True)
ee.Initialize(project=PROJECT)
months = pd.date_range(START, END, freq="MS")[:-1]


# ------------------------------------------------------------------ helpers --
def build_aoi(city_csv):
    c = pd.read_csv(city_csv)
    name = "city_en" if "city_en" in c.columns else "City_Name"
    return ({R: ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]).buffer(R),
                           {"city": str(getattr(r, name))})
                for r in c.itertuples()]) for R in RADII_M}, c, name)


def month_stats(dt, R, AOI):
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


def epoch_totals(epoch, R, AOI):
    """GHSL built-up and population. Two fixes baked in:
       (1) -200 is the no-data fill over water; mask it or coastal airsheds go
           negative (Miami came out at -18.8 million before this).
       (2) reduceRegions names its output after the REDUCER for a single-band
           image, so land_fraction lands in 'mean' - read whichever key exists."""
    built = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{epoch}").select("built_surface")
    pop = ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{epoch}").select("population_count")
    built = built.updateMask(built.gte(0))
    pop = pop.updateMask(pop.gte(0))
    land = pop.mask().unmask(0).rename("land_fraction")

    sums = (built.rename("built_up_m2").addBands(pop.rename("population"))
            .reduceRegions(collection=AOI[R], reducer=ee.Reducer.sum(),
                           scale=100, tileScale=8))
    out = land.reduceRegions(collection=sums, reducer=ee.Reducer.mean(),
                             scale=100, tileScale=8)
    res = {}
    for f in out.getInfo()["features"]:
        p = f["properties"]
        p["land_fraction"] = p.get("land_fraction", p.get("mean"))
        res[p["city"]] = p
    return res


def extract(which):
    tag = which
    final = os.path.join(OUTDIR, f"DUMI_v3_{tag}_complete.csv")
    if os.path.exists(final) and not FORCE:
        print(f"\n{'='*70}\n{tag}: already extracted -> {os.path.basename(final)}"
              f"\n  (set FORCE = True to redo)")
        return

    city_csv = os.path.join(S1, CITY_FILES[which])
    if not os.path.exists(city_csv):
        print(f"\n{'='*70}\n{tag}: SKIPPED - missing {city_csv}")
        return

    AOI, cities, namecol = build_aoi(city_csv)
    print(f"\n{'='*70}")
    print(f"{tag}: {len(cities)} cities | radii {[r//1000 for r in RADII_M]} km | "
          f"{len(months)} months | PBLH = {PBLH_BAND}")
    print(cities[[namecol, "lat", "lon"]].to_string(index=False))

    print("\n[1/3] satellite + meteorology")
    rows, failed = [], []
    for R in RADII_M:
        print(f"  --- {R//1000} km ---")
        for i, m in enumerate(months, 1):
            for attempt in range(3):
                try:
                    rows += [f["properties"]
                             for f in month_stats(m, R, AOI).getInfo()["features"]]
                    break
                except Exception as e:
                    if attempt == 2:
                        failed.append((R // 1000, m.strftime("%Y-%m")))
                        print(f"    FAILED {m:%Y-%m}: {e}")
                    else:
                        time.sleep(5 * (attempt + 1))
            if i % 12 == 0 or i == len(months):
                print(f"    {i:>3}/{len(months)}  rows={len(rows):,}", flush=True)
    if not rows:
        print(f"  {tag}: no data returned - skipping")
        return
    sat = pd.DataFrame(rows)
    sat.to_csv(os.path.join(OUTDIR, f"DUMI_v3_{tag}_satmet.csv"), index=False)

    print("\n[2/3] GHSL urban features")
    urows = []
    for R in RADII_M:
        E = {ep: epoch_totals(ep, R, AOI) for ep in (2020, 2025)}
        neg = sum(1 for v in E[2020].values() if v.get("population", 0) < 0)
        lf = sum(1 for v in E[2020].values() if v.get("land_fraction") is not None)
        print(f"  {R//1000} km: {len(E[2020])} cities | negatives: {neg} | "
              f"land_fraction: {lf}/{len(E[2020])}")
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
    urb.to_csv(os.path.join(OUTDIR, f"urban_v3_{tag}.csv"), index=False)

    print("\n[3/3] merge and validate")
    df = sat.merge(urb, on=["city", "year", "airshed_km"], how="left")
    df["temp_c"] = df["t2m_k"] - 273.15          # the ONE Kelvin conversion
    df["wind_speed_ms"] = (df.wind_u_ms ** 2 + df.wind_v_ms ** 2) ** 0.5
    df["pblh_band"] = PBLH_BAND
    df["s5p_qa_threshold"] = 75                   # applied by GEE at ingest
    df["extraction_date"] = pd.Timestamp.utcnow().date().isoformat()
    df["pipeline_version"] = "v3"
    df["region"] = REGION_TAG[which]
    df["role"] = ROLE[which]

    for c, (lo, hi) in {"temp_c": (-50, 60), "pblh_m": (20, 6000),
                        "no2_vcd_mol_m2": (-1e-3, 5e-3)}.items():
        mn, mx = df[c].min(), df[c].max()
        assert lo <= mn and mx <= hi, f"UNIT VIOLATION {c}: [{mn}, {mx}]"
        print(f"  ok  {c}: [{mn:.4g}, {mx:.4g}]")
    assert (df.population > 0).all(), "negative population - do not proceed"
    assert df.duplicated(["city", "year", "month", "airshed_km"]).sum() == 0

    df.to_csv(final, index=False)
    print(f"\nwrote {final}  ({len(df):,} rows | {df.city.nunique()} cities)")
    print(df[df.airshed_km == 20].groupby("city")
          .agg(pblh_max=("pblh_m", "max"), temp_max=("temp_c", "max"),
               pop=("population", "max"), built_km2=("built_up_km2", "max"))
          .round(0).to_string())
    if failed:
        print(f"  months to re-run: {failed}")
    return df[df.airshed_km == 20].pblh_m.max()


# ---------------------------------------------------------------------- run --
ceilings = {}
for which in REGIONS:
    try:
        p = extract(which)
        if p:
            ceilings[which] = p
    except Exception as e:
        print(f"\n{which} FAILED: {e.__class__.__name__}: {e}")

print("\n" + "=" * 70)
print("PBLH CEILINGS - the constraint that decides the Riyadh claim")
print("  US + China training block reaches 2034 m")
print("  Riyadh requires                 3309 m")
for k, v in ceilings.items():
    print(f"  {k:<14} reaches {v:.0f} m")
if ceilings:
    best = max(list(ceilings.values()) + [2034])
    print(f"\n  new training ceiling: {best:.0f} m")
    print("  Riyadh months above it will be reported by step5d.")
print("\nNext: step5d_assemble_datasets.py")
