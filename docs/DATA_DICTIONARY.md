# Data dictionary

Column definitions for the three assembled datasets. All three share the same
predictor schema; they differ only in whether ground observations are present.

## Keys

| Column | Type | Description |
|---|---|---|
| `city` | text | City name as used throughout the study |
| `year`, `month` | int | Calendar year and month of the observation |
| `airshed_km` | int | Radius of the aggregation circle (20 or 30) |
| `region` | text | `USA`, `CHINA`, `INDIA_TRAIN`, `INDIA_ARID`, `GULF` |

## Satellite

| Column | Unit | Description |
|---|---|---|
| `no2_vcd_mol_m2` | mol m⁻² | Mean tropospheric NO₂ column over the airshed |
| `no2_sd` | mol m⁻² | Standard deviation of the column within the month |
| `sd_ratio` | — | `no2_sd / no2_vcd_mol_m2`, derived |
| `cloud_fraction` | — | Mean cloud fraction, retained as a quality diagnostic |
| `n_obs_no2` | count | TROPOMI granules contributing to the monthly mean |
| `so2_vcd_mol_m2` | mol m⁻² | SO₂ column, extracted but not used (see Sect. 2.2) |

## Meteorology

| Column | Unit | Description |
|---|---|---|
| `pblh_m` | m | MERRA-2 planetary boundary layer height, `PBLH` band |
| `temp_c` | °C | 2 m air temperature, converted once from `T2M` |
| `wind_u_ms`, `wind_v_ms` | m s⁻¹ | 10 m zonal and meridional wind |
| `wind_speed_ms` | m s⁻¹ | Scalar wind speed, derived |

## Urban morphology

| Column | Unit | Description |
|---|---|---|
| `built_up_km2` | km² | GHSL built-up surface summed over the airshed |
| `population` | people | GHSL residential population summed over the airshed |
| `land_fraction` | — | Share of valid (non-water) cells in the airshed |
| `ghsl_interpolated` | bool | True where the year lies between GHSL epochs |
| `ghsl_projected` | bool | True where the epoch is a projection (2025 onward) |

## Ground truth (datasets B and C only)

| Column | Unit | Description |
|---|---|---|
| `ground_no2_ppb` | ppb | Monthly mean over the 10:00–14:00 overpass window |
| `n_stations` | count | Stations contributing to the city-month |

## Model output (`step9_global100_predictions.csv`)

| Column | Unit | Description |
|---|---|---|
| `surface_no2_ppb` | ppb | Estimate, mean over ten seeds |
| `surface_no2_sd_ppb` | ppb | Across-seed standard deviation |
| `surface_no2_ugm3` | µg m⁻³ | Estimate × 1.88 |
| `stage2_weight` | — | Climate gate weight, 0 to 1 |
| `in_domain` | bool | All ten predictors inside the stage-1 training range |
| `expected_rho` | — | Expected column-to-surface correlation from Eq. (7) |
| `confidence` | text | `high`, `moderate`, `low`, `very low` from `expected_rho` |

## Provenance (every record)

| Column | Description |
|---|---|
| `pipeline_version` | Pipeline release that produced the record |
| `pblh_band` | Which MERRA-2 boundary layer variable was used |
| `s5p_qa_threshold` | Retrieval quality threshold applied at ingest |
| `extraction_date` | Date the Earth Engine query was run |
