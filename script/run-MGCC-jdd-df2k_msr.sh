#!/usr/bin/env bash
python -u train.py \
    --gpu_num 0 \
	--train_list ./datasets/train_MSR.txt --valid_list ./datasets/valid_MSR.txt \
    --model MGCC --bias --batch_size 16 --patch_size 64 \
    --valid_freq 500 --save_freq 500 --print_freq 40 --total_epochs 40000 --total_iters 320000 \
    --lr 1e-4  --n_resblocks 4 --channels 64 --block_type rcab\
    #--pretrained_model ./pretrained_model/msr.path \
    

# if run out of memory, lower batch_size down