#!/bin/bash

MODEL="BidLSTM"
current_time=$(date +"%Y%m%d_%H%M%S")
datasets_path="data/aclImdb"
output_path="results/$MODEL/$current_time"
glove_path="data/glove.6B/glove.6B.100d.txt"

python3 main.py \
    --lr 0.001 \
    --batch_size 64 \
    --epochs 20 \
    --model $MODEL \
    --device cuda \
    --max_len 500 \
    --glove_path $glove_path \
    --output_path $output_path \
    --mode train \
    --data_path $datasets_path
