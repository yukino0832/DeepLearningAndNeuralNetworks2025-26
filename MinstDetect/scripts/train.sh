#!/bin/bash

MODEL="LeNet5"
current_time=$(date +"%Y%m%d_%H%M%S")
output_path="results/$MODEL/$current_time"

python3 train.py \
    --epochs 3 \
    --batch_size 32 \
    --lr 0.01 \
    --model $MODEL \
    --mode train \
    --output_path $output_path