#!/bin/bash
#SBATCH -A berzelius-2026-109
#SBATCH --gpus=1
#SBATCH -C thin
#SBATCH -t 23:00:00
#SBATCH -J crystalflow-relax-grid
#SBATCH -o log/%x-%j.out
#SBATCH -e log/%x-%j.err

module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate crystalflow
source .env

MODEL_PATH=/proj/berzelius-2022-164/users/x_behri/CrystalFlow/hydra/singlerun/CSP-mp20-new-eqm
DATASET=mp_20

# null baselines, one per (coord_noise, lattice_noise) pair
for cn in 0.02 0.05 0.1 0.2 0.4; do
  for ln in 0.02 0.05 0.1 0.2 0.4; do
    python scripts/relax_eval.py \
      --model_path $MODEL_PATH \
      --null_baseline \
      --grad-stop 0.05 --min-steps 5 \
      --coord_noise $cn --lattice_noise $ln \
      --label null_c${cn}_l${ln}

    python scripts/compute_metrics.py \
      --root_path $MODEL_PATH \
      --tasks csp \
      --gt_file data/$DATASET/test.csv \
      --label null_c${cn}_l${ln}
  done
done

# model relaxation, same grid
for cn in 0.02 0.05 0.1 0.2 0.4; do
  for ln in 0.02 0.05 0.1 0.2 0.4; do
    python scripts/relax_eval.py \
      --model_path $MODEL_PATH \
      --ode-int-steps 250 --eta 0.03 \
      --coord_noise $cn --lattice_noise $ln \
      --label relax_c${cn}_l${ln}

    python scripts/compute_metrics.py \
      --root_path $MODEL_PATH \
      --tasks csp \
      --gt_file data/$DATASET/test.csv \
      --label relax_c${cn}_l${ln}
  done
done