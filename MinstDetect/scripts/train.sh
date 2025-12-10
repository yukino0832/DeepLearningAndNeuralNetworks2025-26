#!/bin/bash

MODEL="FNN"
current_time=$(date +"%Y%m%d_%H%M%S")
output_path="results/$MODEL/$current_time"

python3 train.py \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.001 \
    --model $MODEL \
    --mode train \
    --output_path $output_path