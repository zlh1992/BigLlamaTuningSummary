#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed run.py \
   --data_path /ossfs/workspace/key_code_01_train_use_v0605.json \
   --data_split 0,10,0 \
   --model_name_or_path /data/Workspace/llamaDemo/ckpt/decapoda-research-llama-7b-hf \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 2048 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 8 \
   --disable_dropout \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --zero_stage 3 \
   --deepspeed \
   --offload \
   --output_dir ./output_0605 \
   &> ./output/training.log
