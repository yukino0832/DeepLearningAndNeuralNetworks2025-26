#!/bin/bash

data_path_need_inference="test_text"
checkpoint_path="results/BidLSTM/20251213_222228/lightning_logs/version_0/checkpoints/best-epoch=09-val_acc=0.8907.ckpt"
glove_path="data/glove.6B/glove.6B.100d.txt"

python3 main.py \
    --batch_size 1 \
    --mode inference \
    --data_path $data_path_need_inference/test.txt \
    --checkpoint_path $checkpoint_path \
    --glove_path $glove_path \
    --output_path $data_path_need_inference/result