#!/bin/bash
#SBATCH -A berzelius-2026-109
#SBATCH --partition=berzelius-cpu
#SBATCH -t 04:00:00
#SBATCH -J wbm-eval
#SBATCH -o log/%x-%j.out
#SBATCH -e log/%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate crystalflow
source .env

MODEL_PATH=/proj/berzelius-2022-164/users/x_behri/CrystalFlow/hydra/singlerun/CSP-mp20-new-eqm
DATASET=wbm
LABEL=wbm-eval

# python scripts/compute_metrics.py \
#   --root_path $MODEL_PATH \
#   --tasks csp \
#   --gt_file /proj/berzelius-2022-164/users/x_behri/wbmdata-crystalflow/groundtruth.csv \
#   --label $LABEL


python scripts/compute_metrics.py \
  --root_path $MODEL_PATH \
  --tasks geo_opt_mbd \
  --gt_file /proj/berzelius-2022-164/users/x_behri/wbmdata-crystalflow/groundtruth.csv \
  --label $LABEL \
  --symprec 1e-2 \
  --njobs 8