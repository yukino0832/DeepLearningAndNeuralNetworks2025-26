#!/bin/bash

datasets_path="data/datasets/"
MODEL="Transformer"
current_time=$(date +"%Y%m%d_%H%M%S")
output_path="results/$MODEL/$current_time"
checkpoint_path="results/Transformer/20251219_174014/lightning_logs/version_0/checkpoints/transformer-epoch=18-val_loss=2.60.ckpt"

python3 main.py \
    --lr 0.0001 \
    --batch_size 128 \
    --epochs 20 \
    --model $MODEL \
    --device cuda \
    --output_path $output_path \
    --mode calculate_bleu \
    --data_path $datasets_path \
    --checkpoint_path $checkpoint_path