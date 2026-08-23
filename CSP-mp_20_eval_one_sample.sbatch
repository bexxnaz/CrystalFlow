#!/bin/bash
#SBATCH -A berzelius-2026-109
#SBATCH --gpus=1
#SBATCH -C thin
#SBATCH --reservation=devel
#SBATCH -t 00:30:00
#SBATCH -J crystalflow-eval-mp20
#SBATCH -o log/%x-%j.out
#SBATCH -e log/%x-%j.err

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate crystalflow
source .env

MODEL_PATH=/proj/berzelius-2022-164/users/x_behri/CrystalFlow/hydra/singlerun/CSP-mp20-new
DATASET=mp_20
LABEL=CSP-mp20-eval

python scripts/evaluate.py \
  --model_path $MODEL_PATH \
  --ode-int-steps 100 \
  --dataset $DATASET \
  --anneal_coords --anneal_slope 5 \
  --label $LABEL

python scripts/compute_metrics.py \
  --root_path $MODEL_PATH \
  --tasks csp \
  --gt_file data/$DATASET/test.csv \
  --label $LABEL