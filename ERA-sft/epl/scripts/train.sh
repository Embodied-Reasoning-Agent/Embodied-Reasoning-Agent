#!/bin/bash
# mkdir -p /work/nvme/bdjz/hchen26/.triton
export TRITON_CACHE_DIR=/u/ry21/.triton

LLM_VERSION=Qwen2.5-VL-3B-Instruct
# LLM_PATH="/u/ry21/scratch/era/sft/checkpoints/eb_man_sft_spatial_grounding_dataset_sampled-lr1e-5-full-e1-bs-16/checkpoints/Qwen2.5-VL-3B-Instruct-sft-stage1"
LLM_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
# LLM_PATH="era-temporary/eb_alfred_sft_stage1_grounding_action_full_planning_randomized"
# LLM_PATH="/u/ry21/scratch/era/sft/checkpoints/eb_man_sft_spatialthinker-1epoch-lr1e-5-full-e1-bs-16/checkpoints/Qwen2.5-VL-3B-Instruct-sft-stage2"
# LLM_PATH="/u/ry21/scratch/era/sft/7b-ckpt/eb_man-stage1-grounding-lr1e-5-full-e1-bs-16/checkpoints/Qwen2.5-VL-7B-Instruct-sft-stage1" # Path to the model checkpoint
# LLM_PATH="/projects/illinois/eng/ece/huanz/era/checkpoints/eb_man_sft_robopoint_grounding_30k-lr1e-5-full-e1-bs-16/checkpoints/Qwen2.5-VL-3B-Instruct-sft-stage1"
# LLM_PATH="/u/ry21/scratch/era/sft/7b-ckpt/eb_alfred_stage1-grounding1+2-lr1e-5-full-e1-bs-16/checkpoints/Qwen2.5-VL-7B-Instruct-sft-stage1"
#LLM_PATH="/projects/illinois/eng/ece/huanz/era/checkpoints/eb_man_sft_robopoint_grounding_30k-lr1e-5-full-e1-bs-16/checkpoints/Qwen2.5-VL-3B-Instruct-sft-stage1"
# LLM_PATH="/projects/illinois/eng/ece/huanz/era/checkpoints/eb_alfred_sft_stage1_grounding_action_full_with_reasoning-stage2-1-2-epoch-mm_traj-lr1e-5-full-e1-bs-16/checkpoints/Qwen2.5-VL-3B-Instruct-sft-stage2/checkpoint-553"
SFT_TASK="stage1" #"stage2"
SAVE_DIR=/u/ry21/scratch/era/sft/checkpoints/eb_man_sft_pipeline_test
IMAGE_FOLDER=/projects/illinois/eng/ece/huanz/era/eb_alfred_sft
# IMAGE_FOLDER=/projects/illinois/eng/ece/huanz/era/eb_man_sft
# IMAGE_FOLDER=/projects/illinois/eng/ece/huanz/era/eb_man_sft/baseline

export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

SFT_DATA_YAML=data/${SFT_TASK}.yaml
SFT_RUN_NAME="${LLM_VERSION}-sft-${SFT_TASK}"
echo "SFT_RUN_NAME: ${SFT_RUN_NAME}"

export CUDA_VISIBLE_DEVICES=0

DISTRIBUTED_ARGS="
    --nproc_per_node 1 \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${RANK} \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
export ACCELERATE_CPU_AFFINITY=0

# printenv

torchrun $DISTRIBUTED_ARGS train.py \
    --deepspeed ./scripts/zero3.json \
    --data_path ${SFT_DATA_YAML} \
    --image_folder ${IMAGE_FOLDER} \
    --model_name_or_path $LLM_PATH \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${SAVE_DIR}/checkpoints/${SFT_RUN_NAME} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
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