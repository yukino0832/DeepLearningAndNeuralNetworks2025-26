#!/bin/bash

MODEL="Transformer"
datasets_path="data/datasets/"
checkpoint_path="results/Transformer/20251219_174014/lightning_logs/version_0/checkpoints/transformer-epoch=18-val_loss=2.60.ckpt"

python3 main.py \
    --model $MODEL \
    --device cuda \
    --mode inference \
    --data_path $datasets_path \
    --checkpoint_path $checkpoint_path \
    --sentence_need_translate "Ein kleines Mädchen spielt mit einem roten Ball im Park."