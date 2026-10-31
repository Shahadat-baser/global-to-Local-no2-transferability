"""
STEP 5B - GULF GROUND TRUTH (OpenAQ, rate-limit safe)
======================================================
    python step5B_gulf_ground_truth.py

Fetches surface NO2 for 5 Gulf cities using the exact sensor IDs found in the
earlier discovery run. Total: 20 sensors.

RATE LIMITING - READ THIS
  The earlier suspension was caused by my discovery script, which paged hourly
  data across ~200 clusters with a 0.3 s delay: tens of thousands of requests
  in one session. This script is deliberately conservative:

    * 20 named sensors only - no searching, no enumeration
    * 1.2 s minimum between requests (~50/min, well under any sane limit)
    * exponential backoff on HTTP 429, and it STOPS after 3 consecutive 429s
    * every response cached to disk, so a re-run costs zero requests
    * hard ceiling of MAX_REQUESTS per session

  Expected total: ~200-600 requests over roughly 10-15 minutes. If you must
  interrupt it, just re-run: cached pages are not re-fetched.

CITIES (all BWh hot desert - the closest available analogues to Riyadh)
  Abu Dhabi, Al Ain, Ruwais, Bida Zayed (UAE) and Hail (Saudi Arabia).
  Note: OpenAQ's locality labels are unreliable - the Hail station is tagged
  "Tekirdag, Turkey" - so cities here are defined by coordinates and explicit
  sensor lists, never by label.

OUTPUT
  GULF_city_monthly_NO2.csv  -> picked up automatically by step5D
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import (STEP1, STEP2, FINAL, CACHE, RAW, FIGDIR,
                   GEE_PROJECT, OPENAQ_API_KEY, require)
import os, sys, time, json, warnings
import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# ------------------------------------------------------------------ settings -
API_KEY = require("OPENAQ_API_KEY")
VERIFY_SSL = True            # set False only behind an HTTPS-inspecting proxy
S1 = str(STEP1)
CITY_CSV   = os.path.join(S1, "gulf_cities_for_extraction.csv")
START, END = "2019-01-01", "2025-12-31"

MIN_INTERVAL_S      = 1.2      # >= 1.2 s between requests
MAX_REQUESTS        = 1500     # hard ceiling for the whole session
MAX_PAGES_PER_SENSOR = 60
OVERPASS_HOURS      = range(10, 15)     # matches TROPOMI ~13:30 local
MIN_HOURS_PER_MONTH = 8
MIN_MONTHS_TO_KEEP  = 12

BASE = "https://api.openaq.org/v3"
os.makedirs(CACHE, exist_ok=True)
os.makedirs(S1, exist_ok=True)
if API_KEY.startswith("PASTE"):
    sys.exit("Set API_KEY first.")

if not VERIFY_SSL:
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    print("!! SSL verification disabled - public data only; re-enable off-network")

S = requests.Session()
S.headers.update({"X-API-Key": API_KEY,
                  "User-Agent": "no2-transferability-research/1.0"})
S.verify = VERIFY_SSL

_state = {"n": 0, "last": 0.0, "consec429": 0}


def get(path, page, **params):
    """One rate-limited, cached GET."""
    key = f"{path.replace('/', '_')}_p{page}.json"
    cp = os.path.join(CACHE, key)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f), True

    if _state["n"] >= MAX_REQUESTS:
        sys.exit(f"\nSession cap of {MAX_REQUESTS} requests reached. "
                 "Re-run later - cached pages will not be re-fetched.")

    for attempt in range(4):
        wait = MIN_INTERVAL_S - (time.time() - _state["last"])
        if wait > 0:
            time.sleep(wait)
        try:
            r = S.get(f"{BASE}/{path}", params={**params, "page": page}, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"      {e.__class__.__name__}, retrying")
            time.sleep(5 * (attempt + 1)); continue
        _state["last"] = time.time(); _state["n"] += 1

        if r.status_code == 200:
            _state["consec429"] = 0
            js = r.json()
            with open(cp, "w") as f:
                json.dump(js, f)
            return js, False
        if r.status_code == 401:
            sys.exit("401 - API key rejected.")
        if r.status_code == 403:
            sys.exit("403 - access forbidden. Stop and contact dev@openaq.org.")
        if r.status_code == 429:
            _state["consec429"] += 1
            if _state["consec429"] >= 3:
                sys.exit("\nThree consecutive rate-limit responses. STOPPING to "
                         "stay within the terms of use. Re-run in an hour; "
                         "cached pages are kept.")
            back = 30 * (attempt + 1)
            print(f"      HTTP 429 - backing off {back}s")
            time.sleep(back); continue
        time.sleep(4 * (attempt + 1))
    return {"results": []}, False


def hourly(sid):
    rows, page, cached = [], 1, 0
    while page <= MAX_PAGES_PER_SENSOR:
        js, was_cached = get(f"sensors/{sid}/hours", page,
                             datetime_from=START, datetime_to=END, limit=1000)
        cached += was_cached
        res = js.get("results", [])
        if not res:
            break
        for m in res:
            p = m.get("period", {}).get("datetimeFrom", {})
            rows.append(dict(t_local=p.get("local"), value=m.get("value")))
        if len(res) < 1000:
            break
        page += 1
    return rows, cached


cities = pd.read_csv(CITY_CSV)
print(f"\n{len(cities)} Gulf cities | {cities.n_sensors.sum()} sensors | "
      f"cache: {CACHE}\n")

recs, report = [], []
for r in cities.itertuples():
    ids = [int(x) for x in str(r.sensor_ids).split(";") if x]
    print(f"{r.city_en} ({r.iso2}) - {len(ids)} sensors")
    got = []
    for sid in ids:
        h, c = hourly(sid)
        for x in h:
            x["sensor_id"] = sid
        got += h
        print(f"    {sid:<10}{len(h):>7} hourly"
              f"{'  (cached)' if c else ''}   [requests used: {_state['n']}]",
              flush=True)
    if not got:
        report.append(dict(city=r.city_en, months=0, kept=False)); continue

    df = pd.DataFrame(got).dropna(subset=["t_local", "value"])
    df["dt"] = pd.to_datetime(df.t_local, errors="coerce", utc=False)
    df = df.dropna(subset=["dt"])
    df = df[df.dt.dt.hour.isin(OVERPASS_HOURS)]
    df["ppb"] = df.value / 1.88                    # ADQCC reports ug/m3
    df = df[(df.ppb >= 0) & (df.ppb < 200)]
    df["year"], df["month"] = df.dt.dt.year, df.dt.dt.month

    st = (df.groupby(["sensor_id", "year", "month"])
            .agg(ppb=("ppb", "mean"), n=("ppb", "size")).reset_index())
    st = st[st.n >= MIN_HOURS_PER_MONTH]
    city = (st.groupby(["year", "month"])
              .agg(ground_no2_ppb=("ppb", "mean"),
                   n_stations=("sensor_id", "nunique"),
                   n_hours=("n", "sum")).reset_index())
    city["city"], city["iso2"] = r.city_en, r.iso2
    city["lat"], city["lon"] = r.lat, r.lon
    city["ground_no2_ugm3"] = city.ground_no2_ppb * 1.88

    keep = len(city) >= MIN_MONTHS_TO_KEEP
    print(f"  -> {len(city)} city-months, mean "
          f"{city.ground_no2_ppb.mean():.1f} ppb   {'KEPT' if keep else 'too sparse'}")
    report.append(dict(city=r.city_en, months=len(city), kept=keep))
    if keep:
        recs.append(city)

print("\n" + "=" * 60)
print(pd.DataFrame(report).to_string(index=False))
print(f"total API requests this session: {_state['n']}")

if not recs:
    sys.exit("\nNo Gulf city met the threshold. step5D will fall back to "
             "20 US + 20 China automatically.")

out = pd.concat(recs, ignore_index=True)
p = os.path.join(S1, "GULF_city_monthly_NO2.csv")
out.to_csv(p, index=False, encoding="utf-8-sig")
print(f"\nwrote {p}  ({len(out)} city-months, {out.city.nunique()} cities)")
print(out.groupby("city").agg(months=("month", "size"), first=("year", "min"),
                              last=("year", "max"),
                              ppb=("ground_no2_ppb", "mean")).round(1).to_string())
print("\nCHECK THE COVERAGE WINDOW. If these cluster in one short period, the")
print("Gulf contribution is a limited-window sample and Methods must say so.")
