# MGCC-pytorch
Pytorch implementation of paper "Joint Image demosaicking and denoising with mutual guidance of color channels"
<p align="center">
  <img height="300" src="figs/MGCC.png">
</p> 

## Installation
```
git clone https://github.com/yongzhangwhu/MGCC-JDD
cd MGCC-JDD  
```
### Requirements
- Python >= 3
- [PyTorch 0.4.1](https://pytorch.org/)
- [Tensorflow](https://www.tensorflow.org/install)  (cpu version is enough, only used for visualization in training)
- opencv-python 
- scipy 
- scikit-image

### Pretrain model
You can download the pretrained models for synthetic and realistic datasets from [here](https://drive.google.com/drive/folders/1jetdV2tXJ8dkg1HLDylhy7e2g9iU1Ilr?usp=sharing).

## Test
- preparation
    - for synthetic datasets
        - add noise by **preprocess.m** using matlab
        - modify --test_noisy_path, --test_gt_path, --sigma, --pretrained_model in **./script/test-MGCC-jdd-df2k.sh**.

    - for MSR dataset
        - generate txt file used for test by **./datasets/generate_image_list.py**.
        - modify --test_datalist, --pretrained_model in **./script/test-MGCC-jdd-df2k_msr.sh**.
- test model
    - test model trained by synthesis datasets 
        ```
        sh ./script/test-MGCC-jdd-df2k.sh  
        ```  
    - test model trained by MSR datasets
        ```
        sh ./script/test-MGCC-jdd-df2k_msr.sh 
        ``` 
## Inference

## Sample results
<p align="center">
  <img height="600" src="figs/JDD_comparison.jpg">
</p> 

## Acknowlegements
