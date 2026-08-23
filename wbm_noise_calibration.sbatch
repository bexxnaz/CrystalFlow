#!/bin/bash
#SBATCH -A berzelius-2026-109
#SBATCH --gpus=1
#SBATCH -C thin
#SBATCH --reservation=devel
#SBATCH -t 02:00:00
#SBATCH -J wbm_noise_calibration
#SBATCH -o log/%x-%j.out
#SBATCH -e log/%x-%j.err

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate crystalflow
source .env

python wbm_noise_calibration.py \
  --initial_csv /proj/berzelius-2022-164/users/x_behri/wbmdata-crystalflow/groundtruth.csv \
  --relaxed_csv /proj/berzelius-2022-164/users/x_behri/wbmdata-crystalflow/test.csv \
  --out wbm_distance_summary.json \
  --niggli true \
  --primitive false \
  --tolerance 0.01 \
  --njobs 16

