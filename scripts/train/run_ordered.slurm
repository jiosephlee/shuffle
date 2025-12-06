#!/bin/bash

#SBATCH --job-name=shuffle_exp
#SBATCH --output=logs/shuffle_exp-%j.out
#SBATCH --time=6:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB

echo "➤ START"

module load anaconda3
module load cuda/12.8.1
module load gcc/13.3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/finetuning"

# Default condition, can be overridden by passing an argument, e.g., sbatch run_experiment.slurm shuffled_para
CONDITION=${1:-ordered}

echo "➤ RUNNING EXPERIMENT FOR CONDITION: $CONDITION"

# Ensure output directory for logs exists
mkdir -p logs

python train.py \
    --condition "$CONDITION" \
    --model_id allenai/Olmo-3-1025-7B \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --packing \
    --gradient_checkpointing \
    --attn_implementation "flash_attention_2"

echo "➤ DONE"
