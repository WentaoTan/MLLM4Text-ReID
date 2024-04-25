#!/bin/bash
DATASET_NAME="Testing"

CUDA_VISIBLE_DEVICES=0,5,6,7 \
python train.py \
--name testing \
--img_aug \
--batch_size 256 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm' \
--num_epoch 30 \
--root_dir /data0/wentao/data/textReID \
--pretrain LuPerson_PEDES \
--nam