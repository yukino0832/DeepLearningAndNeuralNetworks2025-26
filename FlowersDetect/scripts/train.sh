#!/bin/bash

category_names="daisy,dandelion,roses,sunflowers,tulips"
MODEL="ResNet"
current_time=$(date +"%Y%m%d_%H%M%S")
datasets_path="data/datasets"
output_path="results/$MODEL/$current_time"

python3 main.py \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.0001 \
    --model $MODEL \
    --device cuda \
    --input_size 100 \
    --output_path $output_path \
    --mode train \
    --data_path $datasets_path \
    --category_names $category_names
