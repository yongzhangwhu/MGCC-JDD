#!/usr/bin/env bash
python -u test_synthetic.py \
    --gpu_num 0 \
    --model MGCC --bias --n_resblocks 4 --block_type rcab\
    --results_save_path ./test_results_images --metrics_save_path ./test_results_metrics --method MGCC\
    --test_noisy_path ../dataset/noise_AWGN/Kodak_std_10 --test_gt_path ../dataset/Kodak --sigma 10\
    --pretrained_model ./pretrained_model/synthetic.path \

    

# if run out of memory, lower batch_size down

