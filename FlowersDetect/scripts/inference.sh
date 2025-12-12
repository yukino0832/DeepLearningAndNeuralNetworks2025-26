#!/bin/bash

data_path_need_inference="image_test"
checkpoint_path="results/CNN/20251211_204525/lightning_logs/version_0/checkpoints/best-epoch=14-val_acc=0.7188.ckpt"
category_names="daisy,dandelion,roses,sunflowers,tulips"

python3 main.py \
    --batch_size 1 \
    --mode inference \
    --data_path $data_path_need_inference \
    --checkpoint_path $checkpoint_path \
    --output_path $data_path_need_inference/result \
    --category_names $category_names \
    --model "CNN"