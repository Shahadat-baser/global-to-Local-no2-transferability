"""
STEP 5b (alt) - INDIA ARID GROUND TRUTH from CPCB. NO API, NO KEY.
===================================================================
SELF-CONTAINED: downloads AND aggregates. Windows / Linux / macOS.

    pip install pyarrow pandas requests
    python step5b_india_cpcb_direct.py

WHY THIS EXISTS
  My OpenAQ discovery script paged hourly data for ~200 clusters in one session
  with only a 0.3 s delay - tens of thousands of requests - which tripped
  OpenAQ's terms and suspended the account. That was my script's fault.

  This route needs no API and no key: India's Central Pollution Control Board
  data is mirrored as yearly Parquet files on GitHub Releases
  (Vonter/india-cpcb-aqi, ODbL licence). It is also a BETTER source -
  15-minute station records rather than hourly aggregates.

WHAT THE PREVIOUS VERSION GOT WRONG
  The download instructions sat in the docstring instead of in the code, so
  running the file with an empty cpcb/ folder gave "No objects to concatenate".
  Downloading is now part of the script, and if a file is missing it says so
  explicitly instead of failing on an empty list.

DISK / TIME: ~2.7 GB, 15-40 min on a decent connection. Safe to re-run -
existing files are skipped, so an interrupted download resumes.

TIME MATCHING
  CPCB timestamps are UTC. IST = UTC+5:30 and TROPOMI passes ~13:30 local, so
  10:00-14:00 IST = 04:30-08:30 UTC. Same overpass logic as CNEMC and EPA.

UNITS
  CPCB reports NO2 in ug/m3; a ppb column is added (/1.88).

ALREADY RUN FOR YOU - INDIA_city_monthly_NO2.csv is attached. Re-run only if
you want the raw files locally or a different city list.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (STEP1, STEP2, FINAL, CACHE, RAW, FIGDIR,
                   GEE_PROJECT, OPENAQ_API_KEY, require)
import os, sys, glob
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pip install pyarrow")
try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# ------------------------------------------------------------------ settings -
RAWDIR = str(RAW / "cpcb")                      # <<< where parquet goes
OUTCSV = str(STEP1 / "INDIA_city_monthly_NO2.csv")
YEARS  = [2022, 2023, 2024, 2025]
VERIFY_SSL = True           # set False only behind an HTTPS-inspecting proxy

# NW India arid / semi-arid belt (BWh + BSh) - analogues for Saudi cities
ARID = ["Jaipur", "Jodhpur", "Kota", "Ahmedabad", "Gandhinagar",
        "Alwar", "Bikaner", "Ajmer", "Udaipur", "Pali"]

UTC_LO, UTC_HI = 4.5, 8.5          # = 10:00-14:00 IST
MIN_RECORDS_PER_MONTH = 40         # 15-min records ~ 10 hours
MIN_STATIONS = 2
COLS = ["Station ID", "City", "Timestamp", "NO2 (µg/m³)"]
URL = ("https://github.com/Vonter/india-cpcb-aqi/releases/download/"
       "{y}/cpcb-air-quality-{y}.parquet")

os.makedirs(RAWDIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)

if not VERIFY_SSL:
    import warnings
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass


# ----------------------------------------------------------------- download --
def fetch(year):
    path = os.path.join(RAWDIR, f"{year}.parquet")
    if os.path.exists(path) and os.path.getsize(path) > 100_000_000:
        print(f"  {year}: cached ({os.path.getsize(path)/1e6:.0f} MB)")
        return path
    url = URL.format(y=year)
    print(f"  {year}: downloading ...", end="", flush=True)
    try:
        with requests.get(url, stream=True, timeout=180, verify=VERIFY_SSL) as r:
            if r.status_code != 200:
                print(f" HTTP {r.status_code} - skipped")
                return None
            total, done = int(r.headers.get("content-length", 0)), 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk); done += len(chunk)
                    if total and done % (50 << 20) < (1 << 20):
                        print(f" {100*done/total:.0f}%", end="", flush=True)
        print(f" done ({os.path.getsize(path)/1e6:.0f} MB)")
        return path
    except Exception as e:
        print(f" FAILED ({e.__class__.__name__})")
        if os.path.exists(path):
            os.remove(path)          # never leave a truncated file behind
        return None


print(f"[1/2] fetching CPCB yearly files -> {RAWDIR}")
for y in YEARS:
    fetch(y)

files = sorted(glob.glob(os.path.join(RAWDIR, "*.parquet")))
if not files:
    sys.exit(f"\nNo parquet files in {RAWDIR}.\n"
             f"Download at least one manually, e.g.\n  {URL.format(y=2024)}\n"
             f"and save it as {os.path.join(RAWDIR, '2024.parquet')}, then re-run.")
print(f"\n  {len(files)} file(s) ready: {[os.path.basename(f) for f in files]}")


# ---------------------------------------------------------------- aggregate --
print(f"\n[2/2] aggregating (streaming, low memory)")
frames = []
for f in files:
    pf = pq.ParquetFile(f)
    kept = []
    for b in pf.iter_batches(batch_size=500_000, columns=COLS):
        d = b.to_pandas()
        d = d[d.City.isin(ARID)]
        if len(d):
            kept.append(d)
    if kept:
        d = pd.concat(kept, ignore_index=True)
        frames.append(d)
        print(f"  {os.path.basename(f)}: {len(d):,} rows for arid cities")
    else:
        print(f"  {os.path.basename(f)}: no matching cities")

if not frames:
    sys.exit(f"\nFiles read, but no rows for any of {ARID}.\n"
             "Check the City spellings against the CPCB station list.")

raw = pd.concat(frames, ignore_index=True)
raw = raw.rename(columns={"NO2 (µg/m³)": "no2_ugm3", "Station ID": "sid",
                          "City": "city", "Timestamp": "ts"})
raw["ts"] = pd.to_datetime(raw.ts, utc=True, errors="coerce")
raw["no2_ugm3"] = pd.to_numeric(raw.no2_ugm3, errors="coerce")
raw = raw.dropna(subset=["ts", "no2_ugm3"])
raw = raw[(raw.no2_ugm3 >= 0) & (raw.no2_ugm3 < 500)]

h = raw.ts.dt.hour + raw.ts.dt.minute / 60
raw = raw[(h >= UTC_LO) & (h <= UTC_HI)]
ist = raw.ts + pd.Timedelta(hours=5, minutes=30)
raw["year"], raw["month"] = ist.dt.year, ist.dt.month
print(f"\n  overpass-window records: {len(raw):,} | stations {raw.sid.nunique()} "
      f"| {raw.year.min()}-{raw.year.max()}")

st = (raw.groupby(["city", "sid", "year", "month"])
        .agg(no2=("no2_ugm3", "mean"), n=("no2_ugm3", "size")).reset_index())
st = st[st.n >= MIN_RECORDS_PER_MONTH]

city = (st.groupby(["city", "year", "month"])
          .agg(ground_no2_ugm3=("no2", "mean"), n_stations=("sid", "nunique"),
               n_obs=("n", "sum")).reset_index())
city = city[city.n_stations >= MIN_STATIONS]
city["ground_no2_ppb"] = city.ground_no2_ugm3 / 1.88

city.to_csv(OUTCSV, index=False)
print(f"\n  city-months: {len(city)} | cities: {city.city.nunique()}")
print(city.groupby("city").agg(months=("month", "size"), first=("year", "min"),
                               last=("year", "max"),
                               ppb=("ground_no2_ppb", "mean"),
                               stations=("n_stations", "max"))
      .round(1).sort_values("months", ascending=False).to_string())
print(f"\nwrote {OUTCSV}")
