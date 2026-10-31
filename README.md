# Air quality estimation failure in scarcest monitoring networks

**A climate-dependent limit on satellite-to-surface NO₂ inversion**

Code and data pipeline for the manuscript submitted to *Atmospheric Measurement
Techniques*.

> Satellite spectrometers measure tropospheric NO₂ columns, whereas exposure
> assessment requires surface concentrations. Machine-learning inversion is
> widely used to bridge this gap and often reports high predictive skill;
> however, validation is typically conducted within the monitoring network used
> to train the model. The extent to which these models transfer to the
> data-scarce regions where they are most needed remains unquantified. We
> assembled a harmonised database covering 128 cities from 2019 to 2025,
> combining Sentinel-5P TROPOMI NO₂ columns, MERRA-2 meteorology, Global Human
> Settlement Layer urban morphology, and surface observations from four national
> monitoring networks in the United States, China, India, and the Arabian Gulf.
> Under leave-one-city-out validation across 35 temperate and subtropical
> cities, the inversion achieved an R² of 0.570 ± 0.004. The same model and data
> achieved an R² of 0.840 under temporal hold-out, indicating that the gap
> between reported and deployable skill reflects validation design rather than
> model choice. Applied to five hot, arid cities withheld from all model
> development, the inversion performed poorly (R² = −0.83; bias = −7.4 ppb).
> Across all 54 cities with ground observations, the within-city correlation
> between column and surface concentrations declined systematically with mean
> air temperature (r = −0.707, p = 2.3 × 10⁻⁹). A climate-gated, two-stage model
> substantially reduced the arid-region bias to −0.23 ppb but recovered little
> variance (R² = 0.15). Transferability is constrained primarily by
> column-to-surface physics, rather than model capacity or training-set size.

**Keywords** tropospheric NO₂ · Sentinel-5P TROPOMI · surface air quality ·
machine learning · model transferability · spatial cross-validation ·
arid regions · uncertainty quantification
---

## What the study finds

| Result | Value |
|---|---|
| Leave-one-city-out R², 35 temperate and subtropical cities | 0.570 ± 0.004 |
| Same data and model, temporal hold-out | 0.840 |
| Transfer to five held-out hot arid cities | −0.831, bias −7.44 ppb |
| Column-to-surface correlation vs mean temperature, 54 cities | r = −0.707, p = 2.3 × 10⁻⁹ |
| Climate-gated two-stage model on the arid test | +0.148, bias −0.23 ppb |

The gap between 0.840 and 0.570 is a property of the validation design, not of
the model. Skill in a city the model has never seen degrades predictably with
temperature, and the degradation is not removed by adding arid training data.

---

## Quick start

```bash
git clone <repository-url>
cd <repository>

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/paths.py          # prints where data will be read and written
```

Stages 6 to 11 run offline on the archived datasets and reproduce every number
and figure in the paper:

```bash
bash run_pipeline.sh 6 7 8 9 10 11
```

Stages 1 to 5 rebuild the database from source and require network access and
credentials (below). They are not needed to reproduce the published results if
the archived datasets are in place.

---

## Credentials

Two external services are used. Neither key is stored in the repository; both
are read from the environment, and scripts stop with an explanatory message if
they are absent.

```bash
export GEE_PROJECT=your-earth-engine-project-id
export OPENAQ_API_KEY=your-openaq-key
```

On Windows use `setx GEE_PROJECT ...` and reopen the shell.

- **Google Earth Engine** — free for research use; register at
  <https://earthengine.google.com/>. Run `earthengine authenticate` once.
- **OpenAQ** — free key from <https://openaq.org/>. Used only for the Gulf
  network. Please respect the rate limits; `src/step5B_gulf_ground_truth.py`
  caches every response and stops after three consecutive rate-limit replies.

---

## Where data goes

All paths resolve from `src/paths.py`. The default is a `data/` directory inside
the repository; override with `NO2_DATA_ROOT`.

```
data/
  step1_inputs/      city lists, ground-truth CSVs, boundary geojson
  step2_extracted/   Earth Engine output, one file per region
  final_datasets/    datasets A, B, C; fitted models; predictions
  cache/             OpenAQ responses, dissolved country boundaries
  raw/               downloaded archives (CPCB parquet)
figures/
  main/              Figures 1-11
  supplementary/     Figures S1-S10
```

Data files are **not** committed. `.gitignore` excludes `data/` and `figures/`.
The archived datasets are deposited separately (see *Data availability*).

---

## Pipeline

| Stage | Script | Purpose | Network |
|---|---|---|---|
| 1 | `cnemc_download_and_aggregate.py` | Chinese ground truth, 20 cities | yes |
| 2 | `step5b_india_cpcb_direct.py` | Indian ground truth, 15 cities | yes |
| 3 | `step5B_gulf_ground_truth.py` | Gulf ground truth via OpenAQ | yes, key |
| 3b | `step5B_recover_hail.py` | optional, re-examines the excluded Gulf city | cached |
| 4 | `step2e_master_rebuild.py` | Earth Engine: target, US, China | yes, GEE |
| 5 | `step5C_gee_extraction.py` | Earth Engine: Gulf, India train, India test | yes, GEE |
| 6 | `step5d_assemble_datasets.py` | build A, B, C; assert no leakage | no |
| 7 | `step6_eda.py` | exploratory analysis | no |
| 8 | `step7_inversion.py` | inversion under the locked protocol | no |
| 9 | `step8_transferability.py` | transfer tests, uncertainty model | no |
| 10 | `step9_apply_and_shap.py` | apply to 100 cities, SHAP attribution | no |
| 11 | `make_all_figures.py` | all 21 figures at 720 dpi | no |

Run a subset by number: `bash run_pipeline.sh 8 9 10`.

---

## Design constants

These are fixed in `src/paths.py` and are part of the published analysis.
Changing one invalidates comparison with the paper.

| Constant | Value | Note |
|---|---|---|
| `AIRSHED_KM` | 20 | chosen by cross-validation; 30 km also extracted |
| `PBLH_BAND` | `PBLH` | MERRA-2 eddy-diffusion definition, **not** `TCZPBL` |
| `S5P_QA_THRESHOLD` | 75 | applied by `harpconvert` before regridding |
| `OVERPASS_HOURS` | 10:00–14:00 local | matches the 13:30 overpass |
| `PPB_PER_UGM3` | 1.88 | NO₂ at 25 °C, 1013.25 hPa |
| `GATE_LO`, `GATE_HI` | 23, 25 °C | climate gate on city mean temperature |
| `N_SEEDS` | 10 | ensemble members for deployment predictions |

Two of these were the subject of errors during development and are asserted at
runtime. Mixing the two MERRA-2 boundary layer definitions introduces a factor
of two in the variable that controls dilution; supplying temperature in kelvin
to a model fitted in degrees Celsius silently moves every prediction outside the
trained range. `src/step5d_assemble_datasets.py` enforces a unit contract that
halts on either.

---

## Reproducibility

- Hyperparameters are fixed a priori and not tuned against any validation score.
- The validation protocol was fixed before any model was fitted.
- Deployment predictions are the mean over ten seeds; the across-seed standard
  deviation is reported alongside every estimate. A single fit varies by up to
  2 ppb for out-of-domain cities.
- `make_all_figures.py` prints every headline value with the manuscript section
  and figure it appears in, so text and figures can be checked against each
  other after any change.
- Extraction date, QA threshold, boundary layer variable, airshed radius and
  pipeline version are written into every output record.

---

## Data availability

Source data are third-party and subject to their own terms.

| Source | Access |
|---|---|
| Sentinel-5P TROPOMI L3 NO₂, SO₂ | Google Earth Engine, `COPERNICUS/S5P/OFFL/L3_NO2`, `.../L3_SO2` |
| MERRA-2 | Google Earth Engine, `NASA/GSFC/MERRA/flx/2`, `.../slv/2` |
| GHSL P2023A | Google Earth Engine, `JRC/GHSL/P2023A/GHS_BUILT_S`, `.../GHS_POP` |
| US EPA AQS | <https://www.epa.gov/aqs> |
| CNEMC (China) | real-time release, obtained via a public archive |
| CPCB (India) | <https://airquality.cpcb.gov.in>, obtained via a public archive |
| ADQCC (Gulf) | via OpenAQ, <https://openaq.org> |

The three assembled datasets, the fitted models and the 100-city estimates are
archived at: **_DOI to be added on acceptance_**

---

## Citation

Cite both the software and the paper.

**Software** — use the Zenodo *version* DOI for the release you used, not the
concept DOI, so the citation points at the exact code.

```
Baser, S., Bilal, M., Tawabini, B. S., and Koeshidayatullah, A.: Air quality
estimation failure in scarcest monitoring networks: analysis pipeline,
Zenodo [code], https://doi.org/10.5281/zenodo.XXXXXXX, 2026.
```

**Paper** — added on acceptance; see `CITATION.cff`.

`docs/PUBLISHING.md` sets out how to obtain the DOI.

---

## Licensing

Two licences apply, because the repository contains two different kinds of
material.

| Material | Licence | File |
|---|---|---|
| Code in `src/`, `run_pipeline.sh` | MIT | `LICENSE` |
| Derived datasets, models, figures | CC BY 4.0 | `LICENSE-DATA` |
| Source observations | each provider's own terms | see *Data availability* |

The MIT licence text refers throughout to "the Software". That is the standard
legal wording for the material being licensed; here it means the Python scripts
in this repository. It carries no implication about the editor, IDE or operating
system used to write or run them.

MIT is permissive: anyone may use, modify and redistribute the code, including
commercially, provided the copyright notice is retained. CC BY 4.0 is its
equivalent for data and figures, requiring attribution. Neither licence extends
to the source observations, which remain governed by ESA, NASA, the European
Commission and the four monitoring agencies.

---

## Contact

Corresponding author: Shahadat Baser, Department of Geosciences, King Fahd
University of Petroleum and Minerals, Dhahran, Saudi Arabia.

For questions about the code, please open a GitHub issue rather than emailing,
so that answers are visible to others.
