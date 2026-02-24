#!/bin/bash

DATA_ROOT=./datasets/urdf
CKPT_PATH=./checkpoints/last.ckpt
output_dir='./output/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test'
mkdir -p ${output_dir}

CUDA_VISIBLE_DEVICES=0 python eval.py \
  --model_name_or_path ./checkpoints/ShapeLLM_7B_gapartnet_v1.0 \
  --version v1 \
  --vision_tower ./model/ReConV2/cfgs/pretrain/large/openshape.yaml \
  --vision_tower_path ./checkpoints/recon/large.pth \
  --backbone3d_path ./checkpoints/Uni3D/uni3d-b/model.pt \
  --data_root ${DATA_ROOT} \
  --sample_points_num 2048 \
  --with_color True \
  --prompt_token_num 32 \
  --mm_use_pt_start_end False \
  --mm_use_pt_patch_token False \
  --lora_enable True \
  --lora_r 16 \
  --lora_alpha 32 \
  --output_dir ${output_dir} \
  --per_device_eval_batch_size 1 \
  --model_max_length 2048 \
  --dataloader_num_workers 0 \
  --bf16 True \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/mm_projector.bin \
  --load_ckpt_path ${CKPT_PATH}
