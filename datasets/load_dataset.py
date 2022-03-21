import sys
sys.path.append('./datasets')
import numpy as np
import torch
import torch.utils.data as data
import random
from TorchTools.DataTools.Prepro import rgb2raw, data_aug
from model.common import DownsamplingShuffle
import torch.nn.functional as F
import cv2
import os
import torch.nn as nn
from BayerUnifyAug import bayer_aug
from wmad_estimator import *


def extend(x):
    '''
    'Make sure the size of the input is even 
    'input: x,(H, W, C)
    'outpuy:out,the padding result; ex_h, the extension in H; ex_w, the extension in W
    '''
    h, w, _ = x.shape
    ex_h = 0
    ex_w = 0
    out = x
    
    if h%2 != 0:
        ex_h = 1
    if w%2 !=0:
        ex_w = 1
    out = np.pad(x, ((0, ex_h), (0, ex_w),(0,0)), 'symmetric')
    return out, ex_h, ex_w

def estimate_noise(img):
    '''
    'estimate the noise level of img
    'input:img, (B,C,H,W)
    'output:L, the estimated noise level
    '''
    y = img
    if img.max() > 1:
        y  = img / 255

    y = y.sum(dim=1).detach()
    L = Wmad_estimator()(y[:,None])

    if img.max() > 1:
        L *= 255 # scale back to uint8 representation
    return L

####################################################
# load datasets for JDD of synthetic data
####################################################
class Load_Synthetic_train(data.Dataset):
    """
    load training synthetic datasets

    """
    def __init__(self, data_list, patch_size=64, max_noise=16, min_noise=0):
        super(Load_Synthetic_train, self).__init__()
        self.patch_size = patch_size
        self.data_lists = []
        self.max_noise = max_noise/255
        self.min_noise = min_noise/255
        self.raw_stack = DownsamplingShuffle(2)
        
        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            self.data_lists.append(line[0])
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size
        rgb = cv2.cvtColor(cv2.imread(self.data_lists[index]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = data_aug(rgb, mode=np.random.randint(0, 8))
        h, w, c = rgb.shape
        h = h // 2 * 2
        w = w // 2 * 2
        rgb = rgb[0:h, 0:w, :]

        rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, [2, 0, 1]))).float()
        raw = rgb2raw(rgb.clone(), is_tensor=True)

        # crop gt, keep even
        wi = random.randint(0, w - self.patch_size)
        hi = random.randint(0, h - self.patch_size)
        wi = wi - wi%2
        hi = hi - hi%2
        # wi, hi, start point in gt

        rgb = rgb[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
        raw = raw[:, hi: hi + self.patch_size, wi: wi + self.patch_size]

        raw = raw.view(1, 1, raw.shape[-2], raw.shape[-1])
        raw = self.raw_stack(raw)

        
        noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]

        # raw_input + noise
        noise = torch.randn([1, 1, self.patch_size, self.patch_size]).mul_(noise_level)
        raw = raw + self.raw_stack(noise)
        #raw = raw + noise

        # cat noise_map
        noise_map = torch.ones([1, 1, self.patch_size//2, self.patch_size//2])*noise_level
        raw = torch.cat((raw, noise_map), 1)

        data = {}
        data['input'] = torch.clamp(raw[0], 0., 1.)
        data['gt'] = torch.clamp(rgb, 0., 1.)

        return data

class Load_Synthetic_valid(data.Dataset):
    """
    load valid synthetic datasets

    """
    def __init__(self, data_list, patch_size=64, max_noise=16, min_noise=0):
        super(Load_Synthetic_valid, self).__init__()
        self.patch_size = patch_size
        self.data_lists = []
        self.max_noise = max_noise/255
        self.min_noise = min_noise/255
        self.raw_stack = DownsamplingShuffle(2)
        
        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            self.data_lists.append(line[0])
        fin.close()

    def __len__(self):
        return len(self.data_lists)*16

    def __getitem__(self, index):
        # read images, crop size
        index = index%100
        rgb = cv2.cvtColor(cv2.imread(self.data_lists[index]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = data_aug(rgb, mode=np.random.randint(0, 8))
        h, w, c = rgb.shape
        h = h // 2 * 2
        w = w // 2 * 2
        rgb = rgb[0:h, 0:w, :]

        rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, [2, 0, 1]))).float()

        raw = rgb2raw(rgb.clone(), is_tensor=True)

        # crop gt, keep even
        wi = random.randint(0, w - self.patch_size)
        hi = random.randint(0, h - self.patch_size)
        wi = wi - wi%2
        hi = hi - hi%2
        # wi, hi, start point in gt

        rgb = rgb[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
        raw = raw[:, hi: hi + self.patch_size, wi: wi + self.patch_size]

        raw = raw.view(1, 1, raw.shape[-2], raw.shape[-1])
        raw = self.raw_stack(raw)
        
        noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]

        # raw_input + noise
        noise = torch.randn([1, 1, self.patch_size, self.patch_size]).mul_(noise_level)
        raw = raw + self.raw_stack(noise)
        #raw = raw + noise

        # cat noise_map
        noise_map = torch.ones([1, 1, self.patch_size//2, self.patch_size//2])*noise_level
        raw = torch.cat((raw, noise_map), 1)

        data = {}
        data['input'] = torch.clamp(raw[0], 0., 1.)
        data['gt'] = torch.clamp(rgb, 0., 1.)

        return data

class Load_Synthetic_test_MatlabNoisy(data.Dataset):
    """
    load test datasets perturbed by noise 

    """
    def __init__(self, data_gt_path, data_noisy_path, sigma):
        super(Load_Synthetic_test_MatlabNoisy, self).__init__()
        self.data_gt_list = os.listdir(data_gt_path)
        self.data_gt_list.sort()
        self.data_noisy_list = os.listdir(data_noisy_path)
        self.data_noisy_list.sort()
        self.sigma = sigma/255
       
        self.gt_path_and_noisy_path = []
        for i in range(len(self.data_gt_list)):
            self.gt_path_and_noisy_path.append(
                (os.path.join(data_gt_path, self.data_gt_list[i]), os.path.join(data_noisy_path, self.data_noisy_list[i])))
      
        self.raw_stack = DownsamplingShuffle(2)

    def __len__(self):
        return len(self.data_gt_list)
        

    def __getitem__(self, index):
        # read images
        gt_path, noisy_path = self.gt_path_and_noisy_path[index]
        rgb_gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb_noisy = cv2.cvtColor(cv2.imread(noisy_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.

        rgb_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb_gt, [2, 0, 1]))).float()     
        
        rgb_noisy, ex_h, ex_w = np.ascontiguousarray(extend(rgb_noisy.copy())) # padding 
        h, w, c = rgb_noisy.shape                                              
        rgb_noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb_noisy, [2, 0, 1]))).float()
        raw_noisy = rgb2raw(rgb_noisy.clone(), is_tensor=True)
        raw_noisy = raw_noisy.view(1, 1, raw_noisy.shape[-2], raw_noisy.shape[-1])
        raw_noisy = self.raw_stack(raw_noisy)

        
        noise_level = self.sigma 
        noise_map = torch.ones([1, 1, h//2, w//2])*noise_level
        raw_noisy = torch.cat((raw_noisy, noise_map), 1)

        data = {}
        data['input'] = torch.clamp(raw_noisy[0], 0., 1.)
        data['gt'] = torch.clamp(rgb_gt, 0., 1.)

        noisy_name = noisy_path.split('/')[-1]
        return data, noisy_name

####################################################
# load datasets for JDD of linear space data
####################################################
class LoadMSR_dataset_train(data.Dataset):
    """
    load training simulated datasets

    """
    def __init__(self, data_list, patch_size=64):
        super(LoadMSR_dataset_train, self).__init__()
        self.patch_size = patch_size
        self.data_lists = []
        self.raw_stack = DownsamplingShuffle(2)
        
        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip('\n')
            self.data_lists.append(line)
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size
        input_data_path, groundtruth_data_path = self.data_lists[index].split(' ')
       
       ####
        flip_h = True
        flip_w = True
        transpose = True
        if np.random.rand()>0.5:
            flip_h = False
        if np.random.rand()>0.5:
            flip_w = False
        if np.random.rand()>0.5:
            transpose = False
        ####

        rgb_input_original = cv2.imread(input_data_path, -1).astype(np.float32) /65535.
        rgb_groundtruth_orginal = cv2.cvtColor(cv2.imread(groundtruth_data_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        h, w, c = rgb_groundtruth_orginal.shape
        rgb_groundtruth =[]

        rgb_input = bayer_aug(rgb_input_original, flip_h, flip_w, transpose, 'RGGB')
        rgb_input = np.expand_dims(rgb_input, 2)
        for i in range(c):
            gt_channel = bayer_aug(rgb_groundtruth_orginal[:, :, i], flip_h, flip_w, transpose, 'RGGB')
            gt_channel = np.expand_dims(gt_channel, 2)
            rgb_groundtruth.append(gt_channel)
        rgb_groundtruth = np.dstack(rgb_groundtruth)

        rgb_input = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb_input, [2, 0, 1]))).float()
        rgb_groundtruth = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb_groundtruth, [2, 0, 1]))).float()

        # crop gt, keep even
        c, h, w = rgb_input.shape
        wi = random.randint(0, (w - self.patch_size))
        hi = random.randint(0, (h - self.patch_size))
        wi = wi - wi%2
        hi = hi - hi%2
        # wi, hi, start point in gt

        rgb_input = rgb_input[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
        rgb_groundtruth = rgb_groundtruth[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
        
        rgb_input = rgb_input.view(1, 1, rgb_input.shape[-2], rgb_input.shape[-1])
        #estimate noise
        noise_level_estimate = estimate_noise(rgb_input)
        B, C, H, W = rgb_input.size()
        noise_map = torch.ones((B, C, H//2, W//2))*noise_level_estimate
        #print('noise', noise_level_estimate, noise_map.shape)

        rgb_input = self.raw_stack(rgb_input)
        rgb_input = torch.cat([rgb_input, noise_map], 1)
     
        data = {}
        data['input'] = torch.clamp(rgb_input[0], 0., 1.)
        data['gt'] = torch.clamp(rgb_groundtruth, 0., 1.)

        return data

class LoadMSR_dataset_valid(data.Dataset):
    """
    load training simulated datasets

    """
    def __init__(self, data_list):
        super(LoadMSR_dataset_valid, self).__init__()
        self.data_lists = []
        self.raw_stack = DownsamplingShuffle(2)
        
        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip('\n')
            self.data_lists.append(line)
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size
        input_data_path, groundtruth_data_path = self.data_lists[index].split(' ')
        #print('input_data_path:', input_data_path, 'groundtruth_data_path:', groundtruth_data_path)
        rgb_input = cv2.imread(input_data_path, -1).astype(np.float32) / 65535.
        rgb_input = np.expand_dims(rgb_input, 2)
        rgb_groundtruth = cv2.cvtColor(cv2.imread(groundtruth_data_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.

        rgb_input = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb_input, [2, 0, 1]))).float()
        rgb_groundtruth = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb_groundtruth, [2, 0, 1]))).float()
        
        rgb_input = rgb_input.view(1, 1, rgb_input.shape[-2], rgb_input.shape[-1])
        
        #estimate noise
        noise_level_estimate = estimate_noise(rgb_input)
        B, C, H, W = rgb_input.size()
        noise_map = torch.ones((B, C, H//2, W//2))*noise_level_estimate
        #print('noise', noise_level_estimate, noise_map.shape)

        rgb_input = self.raw_stack(rgb_input)
        rgb_input = torch.cat([rgb_input, noise_map], 1)
     
        data = {}
        data['input'] = torch.clamp(rgb_input[0], 0., 1.)
        data['gt'] = torch.clamp(rgb_groundtruth, 0., 1.)

        return data
    
class LoadMSR_dataset_test(data.Dataset):
    """
    load test simulated datasets

    """
    def __init__(self, data_list):
        super(LoadMSR_dataset_test, self).__init__()
        self.data_lists = []
        self.raw_stack = DownsamplingShuffle(2)
        
        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip('\n')
            self.data_lists.append(line)
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size
        input_data_path, groundtruth_data_path = self.data_lists[index].split(' ')
        #print('input_data_path:', input_data_path, 'groundtruth_data_path:', groundtruth_data_path)
        rgb_input = cv2.imread(input_data_path, -1).astype(np.float32) / 65535.
        rgb_input = np.expand_dims(rgb_input, 2)
        rgb_groundtruth = cv2.cvtColor(cv2.imread(groundtruth_data_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.

        rgb_input = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb_input, [2, 0, 1]))).float()
        rgb_groundtruth = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb_groundtruth, [2, 0, 1]))).float()
        
        rgb_input = rgb_input.view(1, 1, rgb_input.shape[-2], rgb_input.shape[-1])
        
        #estimate noise
        noise_level_estimate = estimate_noise(rgb_input)
        B, C, H, W = rgb_input.size()
        noise_map = torch.ones((B, C, H//2, W//2))*noise_level_estimate
        #print('noise', noise_level_estimate, noise_map.shape)

        rgb_input = self.raw_stack(rgb_input)
        rgb_input = torch.cat([rgb_input, noise_map], 1)
     
        data = {}
        data['input'] = torch.clamp(rgb_input[0], 0., 1.)
        data['gt'] = torch.clamp(rgb_groundtruth, 0., 1.)
        noise_name = input_data_path.split('/')[-1]

        return data, noise_name