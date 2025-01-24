#!/bin/bash
DATA=<path to instruction data>
PRETRAINED_CKPT=<path to pretrained ckpt>
OUTPUT_DIR=<your out put dir>
IMAGE_ROOT=<root folder of media file>
MEDIA_MAP=<path to your media file>
deepspeed \
    --include "localhost:0,1,2,3" \
    --master_port 12346 \
    train.py \
    --lora_enable False \
    --split_loading False \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --load $PRETRAINED_CKPT  \
    --version plain \
    --data_path $DATA \
    --image_folder  $IMAGE_ROOT \
    --vision_tower laion/CLIP-ViT-H-14-laion2B-s32B-b79K\
    --vae_image vq-npz \
    --vae_audio vq-npz \
    --media_map $MEDIA_MAP \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --mm_use_gen True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb ${@:1} \
    --output_text \

