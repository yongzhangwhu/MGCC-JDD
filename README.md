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
- Python 3.7
- [PyTorch 1.7.1](https://pytorch.org/)
- [Tensorflow 1.15.0](https://www.tensorflow.org/install) 
- scipy 1.1.0
- scikit-image 0.17.2
- opencv-python

### Pretrain model
You can download the pretrained models for synthetic and realistic datasets from [here](https://drive.google.com/drive/folders/1jetdV2tXJ8dkg1HLDylhy7e2g9iU1Ilr?usp=sharing).

## Test
- preparation
    - for synthetic datasets
        - add noise by **preprocess.m** using matlab for test images.
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
## Train
We train our model on both synthesis datasets([DF2k](https://github.com/xinntao/BasicSR)) and realistic dataset([MSR dataset](https://msropendata.com/datasets/14e910fa-af4a-4a6a-b3b5-fef2f688c923)).

- preparation
    - generate txt file used for training by **./datasets/generate_image_list.py**.
    - modify the training setting in **./script/run-MGCC-jdd-df2k.sh** or **./script/run-MGCC-jdd-df2k_msr.sh**.
- train model
    - on synthetic datasets
      ```
        sh ./script/run-MGCC-jdd-df2k.sh
      ```
    - on realistic dataset
      ```
        sh ./script/run-MGCC-jdd-df2k_msr.sh
      ```
       

## Sample results
<p align="center">
  <img height="700" src="figs/JDD_comparison.jpg">
</p> 

## Acknowlegements
The architecture of our codes are based on [TENet](https://github.com/guochengqian/TENet). 
The pac code is provided by [PACNet](https://github.com/NVlabs/pacnet).
The noise estimation code wmad_estimator.py is provided by [Kokkinos](https://github.com/cig-skoltech/deep_demosaick).
