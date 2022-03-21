#!/usr/bin/env bash
python -u test_msr.py \
    --gpu_num 0 \
    --model MGCC --bias --n_resblocks 4 --block_type rcab\
    --gpu_num 0  --metrics_save_path ./test_results_metrics --method MGCC \
    --results_save_linear_path ./test_results_images_linear/ --results_save_srgb_path ./test_results_images_srgb/\
    --test_datalist ./datasets/test_MSR.txt --gammaparams_path ./MSR_srgb_transform/gammaparams.npy --colortrans_path ./MSR_srgb_transform/colortrans.npy \
    --pretrained_model ./pretrained_model/msr.path \

# if run out of memory, lower batch_size down

