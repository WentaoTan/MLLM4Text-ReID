#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python finetune.py \
--name finetune \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+id+mlm' \
--num_epoch 60 \
--root_dir /data0/wentao/data/textReID \
--finetune The_pretrained_checkpoint
