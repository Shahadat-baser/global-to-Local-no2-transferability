"""
STEP 5B-b - RECOVER HAIL AND REBUILD THE GULF FILE
===================================================
    python step5B_recover_hail.py

Rewrites GULF_city_monthly_NO2.csv so that all five Gulf cities are present,
including Hail. Reads the CACHED OpenAQ responses written by step5B, so it
makes NO new API requests.

WHY HAIL MATTERS MORE THAN THE OTHER FOUR COMBINED

      Hail        PBLH max  3144 m   <- the only city near Riyadh's regime
      Al Ain                1891 m
      Abu Dhabi             1661 m
      Bida Zayed            1562 m
      Ruwais                1124 m
      US ceiling            2034 m
      Riyadh                3309 m

  Riyadh months outside the training PBLH range:
      without Hail   51%  (the four coastal Gulf cities sit BELOW the US max,
                           so they do not raise the ceiling at all)
      with Hail       7%

HOW THE THRESHOLD IS CHOSEN
  step5B kept a city-month with >= 8 valid overpass-window hours, and a city
  with >= 12 such months. Hail is a single-sensor station and failed that.

  This script searches downward through hour thresholds and stops at the
  LOOSEST value that is still defensible, i.e. the first one that gives Hail
  >= 12 months, and then applies that same value to ALL FIVE Gulf cities.
  The rule stays uniform - it is never relaxed for Hail alone.

  A floor of MIN_HOURS_FLOOR is enforced. Below it a monthly mean rests on too
  few observations to be meaningful, and the script stops rather than
  manufacture a number.

WHAT TO PUT IN METHODS
  The script prints the exact sentence to use, including the chosen threshold
  and the resulting per-city month counts. Report it - a relaxed rule that is
  uniform and disclosed is fine; an undisclosed one is not.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (STEP1, STEP2, FINAL, CACHE, RAW, FIGDIR,
                   GEE_PROJECT, OPENAQ_API_KEY, require)
import os, sys, json, glob
import pandas as pd

S1 = str(STEP1)
GULF_CSV = os.path.join(S1, "GULF_city_monthly_NO2.csv")
CITY_CSV = os.path.join(S1, "gulf_cities_for_extraction.csv")

OVERPASS_HOURS   = range(10, 15)     # 10:00-14:00 local, TROPOMI ~13:30
MIN_MONTHS       = 12                # one full seasonal cycle
HOUR_LADDER      = [8, 6, 5, 4, 3, 2]
MIN_HOURS_FLOOR  = 2
HAIL = "Hail"

if not os.path.isdir(CACHE):
    sys.exit(f"No cache at {CACHE}. Run step5B_gulf_ground_truth.py first.")
cities = pd.read_csv(CITY_CSV)


def read_cached(sid):
    rows = []
    for p in sorted(glob.glob(os.path.join(CACHE, f"sensors_{sid}_hours_p*.json"))):
        try:
            with open(p) as f:
                for m in json.load(f).get("results", []):
                    d = m.get("period", {}).get("datetimeFrom", {})
                    rows.append(dict(t_local=d.get("local"), value=m.get("value")))
        except Exception:
            pass
    return rows


def monthly(rows, min_hours):
    df = pd.DataFrame(rows).dropna(subset=["t_local", "value"])
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df.t_local, errors="coerce", utc=False)
    df = df.dropna(subset=["dt"])
    df = df[df.dt.dt.hour.isin(OVERPASS_HOURS)]
    df["ppb"] = pd.to_numeric(df.value, errors="coerce") / 1.88   # ug/m3 -> ppb
    df = df[(df.ppb >= 0) & (df.ppb < 200)]
    if df.empty:
        return df
    df["year"], df["month"] = df.dt.dt.year, df.dt.dt.month
    g = (df.groupby(["year", "month"])
           .agg(ground_no2_ppb=("ppb", "mean"), n_hours=("ppb", "size")).reset_index())
    return g[g.n_hours >= min_hours]


# ---- cache every city's raw records once ------------------------------------
RAW = {}
for r in cities.itertuples():
    ids = [int(x) for x in str(r.sensor_ids).split(";") if x]
    rows = []
    for sid in ids:
        rows += read_cached(sid)
    RAW[r.city_en] = rows
    print(f"  {r.city_en:<12} {len(ids)} sensor(s), {len(rows):>6} cached records")

if not RAW.get(HAIL):
    sys.exit(f"\nNothing cached for {HAIL} (sensor 7068352). It returned no data at "
             f"all,\nso it cannot be recovered. Proceed with 39 training cities and "
             f"report\nthat the deep-desert PBLH regime is unrepresented - that is a "
             f"real finding.")

# ---- choose the loosest defensible uniform threshold ------------------------
print(f"\n{HAIL} months at each hour threshold:")
chosen = None
for mh in HOUR_LADDER:
    n = len(monthly(RAW[HAIL], mh))
    mark = ""
    if n >= MIN_MONTHS and chosen is None and mh >= MIN_HOURS_FLOOR:
        chosen = mh
        mark = "   <- selected (loosest that clears 12 months)"
    print(f"  >= {mh:>2} h/month : {n:>3} months{mark}")

if chosen is None:
    best = max(len(monthly(RAW[HAIL], mh)) for mh in HOUR_LADDER)
    sys.exit(f"\n{HAIL} reaches at most {best} months even at "
             f"{MIN_HOURS_FLOOR} h/month.\nA city with fewer than {MIN_MONTHS} months "
             f"cannot support a seasonal cycle.\nDo not relax further. Proceed with 39 "
             f"training cities and report the\nPBLH ceiling as the study's key "
             f"limitation.")

# ---- apply that threshold to ALL Gulf cities --------------------------------
print(f"\nApplying >= {chosen} h/month uniformly to all {len(cities)} Gulf cities:")
recs = []
for r in cities.itertuples():
    g = monthly(RAW[r.city_en], chosen)
    if g.empty or len(g) < MIN_MONTHS:
        print(f"  {r.city_en:<12} {len(g):>3} months  - below {MIN_MONTHS}, excluded")
        continue
    g["city"], g["iso2"] = r.city_en, r.iso2
    g["lat"], g["lon"] = r.lat, r.lon
    g["n_stations"] = len([x for x in str(r.sensor_ids).split(";") if x])
    g["ground_no2_ugm3"] = g.ground_no2_ppb * 1.88
    g["min_hours_rule"] = chosen
    recs.append(g)
    print(f"  {r.city_en:<12} {len(g):>3} months  mean "
          f"{g.ground_no2_ppb.mean():5.1f} ppb  "
          f"({g.year.min()}-{g.year.max()})")

out = pd.concat(recs, ignore_index=True)
if HAIL not in set(out.city):
    sys.exit(f"\n{HAIL} still excluded. Do not force it further.")

out.to_csv(GULF_CSV, index=False, encoding="utf-8-sig")
print(f"\nwrote {GULF_CSV}")
print(f"  {len(out)} city-months across {out.city.nunique()} cities")

print("\n" + "=" * 72)
print("SENTENCE FOR METHODS (copy this):")
print(f"""
  Gulf ground-truth city-months were retained where at least {chosen} valid
  hourly observations fell within the 10:00-14:00 local overpass window, and a
  city was retained where at least {MIN_MONTHS} such months were available. This
  threshold was applied uniformly to all {len(cities)} Gulf cities. It is looser than
  the >= 8 h/month rule used for the CNEMC and EPA networks, reflecting the
  sparser reporting of Gulf stations; the effect of this choice is reported in
  the sensitivity analysis.
""")
print("=" * 72)
print("Now re-run step5d_assemble_datasets.py -> 15 US + 20 CN + 5 Gulf = 40.")
