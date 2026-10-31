"""
Central configuration for paths and credentials.

Every script in this pipeline imports from here, so there is exactly one place
to edit when the repository is cloned. Nothing else contains an absolute path.

Resolution order for each setting:
  1. an environment variable, if set
  2. the default below, relative to the repository root

So the pipeline runs unchanged after a clone, and a different data location can
be supplied without editing any file:

    export NO2_DATA_ROOT=/scratch/no2
    export GEE_PROJECT=my-earth-engine-project
    export OPENAQ_API_KEY=...

CREDENTIALS ARE NEVER STORED IN THIS FILE. Both are read from the environment
and the scripts stop with a clear message if they are missing.
"""
import os
from pathlib import Path

# repository root, inferred from this file's location
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- data tree --
DATA_ROOT = Path(os.environ.get("NO2_DATA_ROOT", ROOT / "data"))

STEP1 = DATA_ROOT / "step1_inputs"      # city lists, ground-truth CSVs
STEP2 = DATA_ROOT / "step2_extracted"   # Earth Engine output
FINAL = DATA_ROOT / "final_datasets"    # datasets A, B, C and model output
CACHE = DATA_ROOT / "cache"             # OpenAQ responses, dissolved boundaries
RAW = DATA_ROOT / "raw"                 # downloaded archives (CPCB parquet etc.)

FIGDIR = Path(os.environ.get("NO2_FIG_DIR", ROOT / "figures"))

for _d in (STEP1, STEP2, FINAL, CACHE, RAW, FIGDIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- credentials --
GEE_PROJECT = os.environ.get("GEE_PROJECT", "")
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")


def require(name: str) -> str:
    """Return a credential or stop with an actionable message."""
    value = {"GEE_PROJECT": GEE_PROJECT, "OPENAQ_API_KEY": OPENAQ_API_KEY}[name]
    if not value:
        raise SystemExit(
            f"\n{name} is not set.\n"
            f"  export {name}=...        (Linux, macOS)\n"
            f"  setx {name} ...          (Windows, then reopen the shell)\n"
            f"See README.md, section 'Credentials'.\n")
    return value


# ------------------------------------------------------------ fixed design --
# These are analysis constants, not user settings. Changing one invalidates the
# comparison with the published results.
AIRSHED_KM = 20                  # selected by cross-validation; 30 km also extracted
PBLH_BAND = "PBLH"               # MERRA-2 eddy-diffusion definition, not TCZPBL
S5P_QA_THRESHOLD = 75            # applied by harpconvert before regridding
OVERPASS_HOURS = range(10, 15)   # 10:00-14:00 local, matches the 13:30 overpass
PPB_PER_UGM3 = 1.88              # NO2 at 25 C, 1013.25 hPa
GATE_LO, GATE_HI = 23.0, 25.0    # climate gate on city mean temperature
N_SEEDS = 10                     # ensemble members for deployment predictions

START_DATE, END_DATE = "2019-01-01", "2026-01-01"

if __name__ == "__main__":
    print(f"repository root : {ROOT}")
    print(f"data root       : {DATA_ROOT}")
    for n, d in [("step1", STEP1), ("step2", STEP2), ("final", FINAL),
                 ("cache", CACHE), ("raw", RAW), ("figures", FIGDIR)]:
        print(f"  {n:<8} {d}")
    print(f"\nGEE_PROJECT    : {'set' if GEE_PROJECT else 'NOT SET'}")
    print(f"OPENAQ_API_KEY : {'set' if OPENAQ_API_KEY else 'NOT SET'}")
