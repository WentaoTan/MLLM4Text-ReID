#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0,2,3 \
python finetune.py \
--name 60w \
--img_aug \
--batch_size 192 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm' \
--num_epoch 30 \
--root_dir /data0/wentao/data/textReID \
--finetune LuPerson_PEDES