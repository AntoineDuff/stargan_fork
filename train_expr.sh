#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-jlalonde
#SBATCH --gres=gpu:v100:1             # Number of GPU(s) per node
#SBATCH --cpus-per-task=8          # CPU cores/threads
#SBATCH --mem=16000M            # memory per node
#SBATCH --time=0-24:00           # time (DD-HH:MM)

EXPR="baseline_128"
echo "🚀 $EXPR"

nvidia-smi

echo "👉 Activating environment"
cd $SCRATCH/stargan_fork
source load_slurm_modules.sh
source $HOME/stargan-v2-env/bin/activate

echo "👉 Starting training"
# if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
#     echo "Please supply an array task ID"
#     exit 1 
# fi
if [ -z "$EXPR" ]; then
    echo "Please supply an experiment name"
    exit 1 
fi

export MKL_NUM_THREADS=1 # *** important else scipy `sqrtm` takes ∞ time

echo "Custom args: $@"

python main.py --mode train --num_domains 3 --w_hpf 0 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
               --train_img_dir data/afhq/train \
               --val_img_dir data/afhq/val \
               --img_size 128 \
               --checkpoint_dir expr/"$EXPR"/checkpoints/afhq \
               --result_dir expr/"$EXPR"/results/afhq \
               --sample_dir expr/"$EXPR"/samples/afhq \
               --eval_dir expr/"$EXPR"/eval/afhq \
               --wing_path expr/"$EXPR"/checkpoints/wing.ckpt \
               --lm_path expr/"$EXPR"/checkpoints/celeba_lm_mean.npz \
               --notes_path expr/"$EXPR" \
               --use_mean_shift False \
               --use_mlp False
               "$@"