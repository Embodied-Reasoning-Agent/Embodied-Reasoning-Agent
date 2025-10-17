#!/bin/bash

LLM_VERSION=Qwen2.5-VL-3B-Instruct
LLM_PATH="Qwen/Qwen2.5-VL-3B-Instruct"  
SFT_TASK="stage"
SAVE_DIR=PATH_TO_SAVE_DIR
IMAGE_FOLDER=PATH_TO_IMAGE_FOLDER

SFT_DATA_YAML=data/${SFT_TASK}.yaml
SFT_RUN_NAME="${LLM_VERSION}-sft-${SFT_TASK}"
echo "SFT_RUN_NAME: ${SFT_RUN_NAME}"

export CUDA_VISIBLE_DEVICES=0,1

DISTRIBUTED_ARGS="
    --nproc_per_node 2 \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
export ACCELERATE_CPU_AFFINITY=0

printenv

torchrun $DISTRIBUTED_ARGS train.py \
    --deepspeed ./scripts/zero3.json \
    --data_path ${SFT_DATA_YAML} \
    --image_folder ${IMAGE_FOLDER} \
    --model_name_or_path $LLM_PATH \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${SAVE_DIR}/checkpoints/${SFT_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --freeze_visual_encoder True