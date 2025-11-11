#!/bin/bash

# 定义GPU
gpu=0
# 定义PLM模型和对应的 d_model
# 你可以根据你的硬件和需求选择不同的模型
# Qwen/Qwen2.5-0.5B-Instruct -> d_model=896
# Qwen/Qwen2.5-1.5B-Instruct -> d_model=1536
# Qwen/Qwen2.5-7B-Instruct -> d_model=3584
PLM_PATH="Qwen/Qwen2.5-1.5B-Instruct"
D_MODEL=1536

# LoRA 参数
LORA_R=8
LORA_ALPHA=16

# 遍历5个split
for split in 1 2 3 4 5
do
echo "Running split $split with $PLM_PATH"

# 运行 classification.py
python classification.py \
    --model istsplm \
    --task 'P12' \
    --state 'qwen_lora_ctrope' \
    --split $split \
    --seed 0 \
    --gpu $gpu \
    --batch_size 6 \
    --lr 1e-4 \
    --epoch 20 \
    --patience 20 \
    --n_classes 2 \
    --max_len -1 \
    --dropout 0.1 \
    --semi_freeze \
    --sample_rate 1 \
    \
    --plm_path $PLM_PATH \
    --d_model $D_MODEL \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA
    
done
