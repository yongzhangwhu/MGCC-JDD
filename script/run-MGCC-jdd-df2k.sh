#!/usr/bin/env bash
python -u train.py \
	--gpu_num 0 \
    --train_list ./datasets/train_df2k.txt --valid_list ./datasets/valid_div2k.txt \
    --model MGCC --bias  --max_noise 20 --min_noise 0 --batch_size 4 --patch_size 128 \
    --valid_freq 2000 --save_freq 2000 --print_freq 80 --total_epochs 4000 --total_iters 640000 \
    --lr 1e-4  --n_resblocks 4 --channels 64 --block_type rcab\
    #--pretrained_model ./checkpoints/checkpoints-demo-dn-df2kx6-6-6-64-2-rrdb/demo-dn-df2kx6-6-6-64-2-rrdb_checkpoint_240.0k.path \
    

# if run out of memory, lower batch_size down
