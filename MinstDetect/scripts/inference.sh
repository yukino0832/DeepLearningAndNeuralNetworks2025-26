#!/bin/bash

data_path_need_inference="image_test"
checkpoint_path="results/FNN/20251209_194518/lightning_logs/version_0/checkpoints/best-epoch=12-val_loss=0.0651.ckpt"

python3 train.py \
    --batch_size 1 \
    --mode inference \
    --data_path $data_path_need_inference \
    --checkpoint_path $checkpoint_path \
    --output_path $data_path_need_inference/result