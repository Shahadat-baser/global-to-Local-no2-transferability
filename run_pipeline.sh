#!/usr/bin/env bash
# Run the full pipeline end to end.
#   bash run_pipeline.sh            all stages
#   bash run_pipeline.sh 7 8 9      selected stages only
#
# Stages 1-5 need network access and credentials; 6-10 run offline on the
# assembled datasets. A stage whose output already exists is skipped unless
# FORCE=1 is set.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"

declare -A STAGE=(
  [1]="src/cnemc_download_and_aggregate.py:Chinese ground truth (CNEMC)"
  [2]="src/step5b_india_cpcb_direct.py:Indian ground truth (CPCB)"
  [3]="src/step5B_gulf_ground_truth.py:Gulf ground truth (OpenAQ) - needs OPENAQ_API_KEY"
  [4]="src/step2e_master_rebuild.py:Earth Engine extraction, target + US + China - needs GEE_PROJECT"
  [5]="src/step5C_gee_extraction.py:Earth Engine extraction, Gulf + India"
  [6]="src/step5d_assemble_datasets.py:assemble datasets A, B, C with leakage assertions"
  [7]="src/step6_eda.py:exploratory analysis"
  [8]="src/step7_inversion.py:inversion under the locked protocol"
  [9]="src/step8_transferability.py:transferability and the uncertainty model"
  [10]="src/step9_apply_and_shap.py:apply to the 100 target cities, SHAP"
  [11]="src/make_all_figures.py:all manuscript and supplement figures"
)
ORDER=(1 2 3 4 5 6 7 8 9 10 11)
TODO=("${@:-${ORDER[@]}}")

for s in "${TODO[@]}"; do
  entry="${STAGE[$s]:-}"
  [[ -z "$entry" ]] && { echo "unknown stage: $s"; exit 1; }
  script="${entry%%:*}"; label="${entry#*:}"
  echo
  echo "=============================================================="
  echo " stage $s  -  $label"
  echo "=============================================================="
  python3 "$script"
done
echo
echo "done."
