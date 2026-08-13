#!/bin/bash
#SBATCH -A berzelius-2026-109
#SBATCH --gpus=1
#SBATCH -C thin
#SBATCH -t 23:00:00
#SBATCH -J crystalflow-eval-ms-mpts52
#SBATCH -o log/%x-%j.out
#SBATCH -e log/%x-%j.err

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate crystalflow
source .env

MODEL_PATH=/proj/berzelius-2022-164/users/x_behri/CrystalFlow/hydra/singlerun/CSP-mpts52
DATASET=mpts_52
LABEL=CSP-mpts52-multiple-samples-eval

python scripts/evaluate.py \
  --model_path $MODEL_PATH \
  --ode-int-steps 100 \
  --dataset $DATASET \
  --anneal_coords --anneal_slope 5 \
  --num_evals 20 \
  --label $LABEL

python scripts/compute_metrics.py \
  --root_path $MODEL_PATH \
  --tasks csp \
  --gt_file data/$DATASET/test.csv \
  --multi_eval \
  --label $LABEL


