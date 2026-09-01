#!/bin/bash
#SBATCH -A berzelius-2026-109
#SBATCH --gpus=1
#SBATCH -C thin
#SBATCH -t 07:00:00
#SBATCH -J wbmtest_new
#SBATCH -o log/%x-%j.out
#SBATCH -e log/%x-%j.err


module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate crystalflow
source .env

MODEL_PATH=/proj/berzelius-2022-164/users/x_behri/CrystalFlow/hydra/singlerun/CSP-mp20-new-eqm
DATASET=wbm
LABEL=wbmtest_without_anneal


# python scripts/relax_wbm.py --model_path $MODEL_PATH \
#   --ode-int-steps 250 --eta 0.03 --grad-stop-coord 0.03 --grad-stop-lattice 0.02 --min-steps 5 --dataset $DATASET \
#   --anneal_coords --anneal_slope 5 \
#   --label $LABEL 

# python scripts/relax_wbm.py \
#   --model_path $MODEL_PATH \
#   --dataset $DATASET \
#   --ode-int-steps 250 \
#   --eta 0.03 \
#   --sampler gd \
#   --grad-stop 0.05 --min-steps 5 \
#   --label $LABEL \



# python scripts/relax_wbm.py \
#   --model_path $MODEL_PATH \
#   --dataset $DATASET \
#   --ode-int-steps 250 \
#   --eta 0.03 \
#   --sampler gd \
#   --grad-stop-coord 0.03 --grad-stop-lattice 0.02 --min-steps 5 \
#   --anneal_coords --anneal_slope 5 \
#   --label $LABEL \

# python scripts/compute_metrics.py \
#   --root_path $MODEL_PATH \
#   --tasks test_sym \
#   --gt_file /proj/berzelius-2022-164/users/x_behri/wbmdata-crystalflow/groundtruth.csv \
#   --label $LABEL \
#   --njobs 8

python scripts/compute_metrics.py \
  --root_path $MODEL_PATH \
  --tasks geo_opt_mbd \
  --gt_file /proj/berzelius-2022-164/users/x_behri/wbmdata-crystalflow/groundtruth.csv \
  --label $LABEL \
  -j 8 \
  --symprec 1e-2

