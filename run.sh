#!/bin/bash

# Dataset
dataset_name=tiny
context_length=256
vocab_size=50257

# transformer architecture
transformer_type="loop" # gpt | loop | select | loop_concat
emb_dim=768
n_heads=12
n_layers=1
drop_rate=0.1
## Feedback hyperparameters
n_iter=12
## SFTransformer hyperparameters
select_dim=512
select_heads=4
temperature=0 

# training hyperparameters
epochs=15
peak_lr=0.001
weight_decay=0.1
batch_size=4
use_wandb=False
folder_to_save="checkpoints"
cosine_annealing=False
num_workers=0
eval_freq=5
eval_iter=1

python3 main.py \
    --vocab_size $vocab_size \
    --context_length $context_length \
    --emb_dim $emb_dim \
    --n_heads $n_heads \
    --n_layers $n_layers \
    --drop_rate $drop_rate \
    --batch_size $batch_size \
    --n_iter $n_iter \
    --select_dim $select_dim \
    --select_heads $select_heads \
    --temperature $temperature \
    --epochs $epochs \
    --dataset_name $dataset_name \
    --transformer_type $transformer_type \
    --peak_lr $peak_lr \
    --weight_decay $weight_decay \
    --use_wandb $use_wandb \
    --folder_to_save $folder_to_save \
    --num_workers $num_workers \
    --eval_freq $eval_freq \
    --eval_iter $eval_iter \
    --cosine_annealing $cosine_annealing
