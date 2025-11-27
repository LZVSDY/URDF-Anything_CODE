#!/bin/bash
home_dir=xxx

LLM_VERSION=./checkpoints/ShapeLLM_7B_general_v1.0

export TZ='Asia/Shanghai'
CURRENT_TIME=$(date +"%m%d_%H%M")
echo "Current time: $CURRENT_TIME"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


python train_lightning.py \
    --lora_enable True --lora_r 8 --lora_alpha 16 --mm_projector_lr 2e-5 \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --vision_tower ./model/ReConV2/cfgs/pretrain/large/openshape.yaml \
    --vision_tower_path ./checkpoints/recon/large.pth \
    --backbone3d_path ./checkpoints/Uni3D/uni3d-b/model.pt \
    --sample_points_num 2048 \
    --with_color True \
    --occlusion False \
    --prompt_token_num 32 \
    --with_ape True \
    --with_local True \
    --with_global True \
    --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_pt_start_end False \
    --mm_use_pt_patch_token False \
    --group_by_modality_length True \
    --output_dir ./output/$LLM_VERSION-lora/${CURRENT_TIME} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --evaluation_strategy "no" \
    --num_train_epochs 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 False \
    --model_max_length 2048 \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to wandb \
    --predict_type all_parameters 
      