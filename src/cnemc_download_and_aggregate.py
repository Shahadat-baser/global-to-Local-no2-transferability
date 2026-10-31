"""
CNEMC China ground-station NO2 -> monthly city means.

Source : https://github.com/HeQinWill/CNEMC  (mirrors air.cnemc.cn:18007)
Period : 2022-01-01 .. 2025-12-31
Hour   : 13:00 Beijing time = Sentinel-5P/TROPOMI overpass window (~13:30 local),
         so the ground value is temporally matched to the satellite retrieval
         rather than being a diurnal average.
Units  : CNEMC reports NO2 in ug/m3. ppb column added (/1.88).

STEP A - download (run once; ~470 MB, 1461 files):
    python -c "
import datetime as dt
d=dt.date(2022,1,1); e=dt.date(2025,12,31); u=[]
while d<=e:
    s=d.isoformat()
    u.append(f'https://raw.githubusercontent.com/HeQinWill/CNEMC/main/Archive/{s}/{s}T13.csv')
    d+=dt.timedelta(days=1)
open('urls.txt','w').write(chr(10).join(u))"
    mkdir cnemc_raw
    xargs -P 24 -n 1 -a urls.txt sh -c 'f=$(echo "$0"|sed "s#.*/##"); curl -sf "$0" -o cnemc_raw/$f'

STEP B - aggregate: python cnemc_download_and_aggregate.py

NOTE: the archive uses TWO timestamp formats. Pre-Nov-2023 files use
'2022-03-15T13:00:00'; later files use '2023-11-20T1300'. Parsing only one
silently discards ~2.3 M rows (all of 2022 and most of 2023).
"""
import os, glob, numpy as np, pandas as pd

RAW = "/home/claude/cnemc_raw"
files = sorted(glob.glob(os.path.join(RAW, "*.csv")))
print(f"{len(files)} daily files")

keep = ["timepoint", "stationcode", "longitude", "latitude", "area", "no2", "so2"]
frames = []
bad = 0
for f in files:
    try:
        d = pd.read_csv(f, usecols=lambda c: c in keep, low_memory=False)
        if "no2" not in d.columns:
            bad += 1; continue
        frames.append(d)
    except Exception:
        bad += 1

raw = pd.concat(frames, ignore_index=True)
print(f"unreadable files: {bad}")
print(f"raw station-hours: {len(raw):,}")

raw["timepoint"] = raw.timepoint.astype(str)
# archive uses TWO formats: '2022-03-15T13:00:00' (pre Nov-2023) and '2023-11-20T1300'
dt_a = pd.to_datetime(raw.timepoint, format="%Y-%m-%dT%H:%M:%S", errors="coerce")
dt_b = pd.to_datetime(raw.timepoint, format="%Y-%m-%dT%H%M", errors="coerce")
raw["dt"] = dt_a.fillna(dt_b)
print(f"  parsed old format: {dt_a.notna().sum():,} | new format: {dt_b.notna().sum():,}")
raw = raw.dropna(subset=["dt", "no2", "latitude", "longitude"])
raw["year"] = raw.dt.dt.year
raw["month"] = raw.dt.dt.month

# CNEMC reports NO2 in ug/m3
raw["no2"] = pd.to_numeric(raw.no2, errors="coerce")
raw = raw[(raw.no2 >= 0) & (raw.no2 < 500)]
print(f"after QC: {len(raw):,} station-hours | "
      f"{raw.stationcode.nunique():,} stations | {raw.area.nunique()} prefectures")
print(f"period: {raw.dt.min().date()} -> {raw.dt.max().date()}")

# ---- station -> city (prefecture) monthly means -----------------------------
st = (raw.groupby(["area", "stationcode", "year", "month"])
         .agg(no2_ugm3=("no2", "mean"), lat=("latitude", "mean"),
              lon=("longitude", "mean"), n_hours=("no2", "size"))
         .reset_index())
st = st[st.n_hours >= 15]          # >=15 valid overpass-hours in the month

city = (st.groupby(["area", "year", "month"])
          .agg(ground_no2_ugm3=("no2_ugm3", "mean"),
               lat=("lat", "mean"), lon=("lon", "mean"),
               n_stations=("stationcode", "nunique"),
               n_hours=("n_hours", "sum"))
          .reset_index())
city = city[city.n_stations >= 3]  # >=3 stations for a defensible city mean
city["ground_no2_ppb"] = city.ground_no2_ugm3 / 1.88

print(f"\ncity-months: {len(city):,} | cities: {city.area.nunique()}")
print(f"NO2 mean {city.ground_no2_ugm3.mean():.1f} ug/m3 "
      f"({city.ground_no2_ppb.mean():.1f} ppb)")

cov = city.groupby("area").size().sort_values(ascending=False)
print(f"\ncities with >=36 months: {(cov >= 36).sum()}")
print(f"cities with >=24 months: {(cov >= 24).sum()}")

out = "/home/claude/CNEMC_city_monthly_NO2.csv"
city.sort_values(["area", "year", "month"]).to_csv(out, index=False)
print(f"\nwrote {out}  ({os.path.getsize(out)/1024:.0f} KB)")
print("\nTop 15 cities by mean NO2:")
top = city.groupby("area").ground_no2_ugm3.agg(["mean", "size"]).query("size>=24")
print(top.sort_values("mean", ascending=False).head(15).to_string(
    float_format=lambda x: f"{x:.1f}"))
