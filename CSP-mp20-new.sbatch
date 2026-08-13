#!/bin/bash
#SBATCH -A berzelius-2026-109
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 3-00:00:00
#SBATCH -J crystalflow-csp-3024bec
#SBATCH -o log/%x-%j.out
#SBATCH -e log/%x-%j.err


module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate crystalflow
source .env

HYDRA_FULL_ERROR=1 python diffcsp/run.py \
  data=mp_20 data.train_max_epochs=3000 \
  ~data.conditions \
  model=flow_polar \
  ~model.conditions \
  ~model.cost_type \
  ~model.decoder.smooth \
  ~model.decoder.pred_type \
  optim.optimizer.lr=1e-3 \
  optim.optimizer.weight_decay=0 \
  optim.lr_scheduler.factor=0.6 \
  +model.lattice_polar_sigma=0.1 \
  model.cost_coord=10 model.cost_lattice=1 \
  model.decoder.num_freqs=256 \
  model.decoder.rec_emb=sin model.decoder.num_millers=8 \
  +model.decoder.na_emb=0 \
  model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
  train.model_checkpoints.save_top_k=5 \
  train.model_checkpoints.save_last=true \
  logging.wandb.mode=offline \
  logging.wandb.project=crystalflow-gridtest \
  logging.val_check_interval=1 \
  expname=CSP-mp20-new 
