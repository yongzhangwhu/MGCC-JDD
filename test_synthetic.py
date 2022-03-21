import os
import argparse
import shutil
import importlib
import numpy as np
import time

import torch
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm

from datasets.load_dataset import Load_Synthetic_test_MatlabNoisy
from model.common import print_model_parm_nums
import torch.nn as nn
from TorchTools.ArgsTools.test_args import TestArgs
from TorchTools.LogTools.logger_tensorboard import Tf_Logger
from TorchTools.LossTools.loss import C_Loss, VGGLoss
from TorchTools.LossTools.metrics import PSNR, AverageMeter
import utils


def main():
    ###############################################################################################
    # args parse
    parser = argparse.ArgumentParser(description='PyTorch implementation of MGCC for test')
    parsers = TestArgs()
    args = parsers.initialize(parser)
    parsers.print_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    noisy_dataset_name = args.test_noisy_path.split('/')[-1]
    if not os.path.exists(args.results_save_path):
        os.mkdir(args.results_save_path) 
    if not os.path.exists(args.metrics_save_path):
        os.mkdir(args.metrics_save_path)  

    ###############################################################################################
    # load netwoork
    print('===> Loading the network ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(device)
    #print(model)
    print_model_parm_nums(model)
   
    ###############################################################################################
    # load checkpoints
    best_psnr = 0
    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            print("=====> loading checkpoint '{}'".format(args.pretrained_model))
            checkpoint = torch.load(args.pretrained_model)
            best_psnr = checkpoint['best_psnr']
            model.load_state_dict(checkpoint['state_dict'])
            print("The pretrained_model is at checkpoint {}, and it's best loss is {}."
                  .format(checkpoint['iter'], best_psnr))
        else:
            print("=====> no checkpoint found at '{}'".format(args.pretrained_model))
    
    ###############################################################################################
    # create output path
    pr_img_path = os.path.join(args.results_save_path, args.method + '_' + noisy_dataset_name)
    if not os.path.exists(pr_img_path):
        os.mkdir(pr_img_path)      
    pr_metrics_path = os.path.join(args.metrics_save_path, args.method + '_' + noisy_dataset_name + '.txt')
    metric_file = open(pr_metrics_path, 'w')
    metric_file.write("{0:<10}\t{1:<10}\t{2:<10}\n".format('img_name', 'psnr', 'ssim'))

    print('===> Creating dataloader...')
    test_set = Load_Synthetic_test_MatlabNoisy(args.test_gt_path, args.test_noisy_path, args.sigma)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False, pin_memory=True)

    psnr, ssim = test(test_loader, model, args, device, pr_img_path, metric_file)
    metric_file.write("{0:<10}\t{1:<10.8s}\t{2:<10.8s}\n".format(noisy_dataset_name, str(psnr.item()), str(ssim.item())))
    metric_file.close()

############################################################################################
#
#   test functions
#
############################################################################################
def test(test_loader, model, args, device, pr_img_path, metric_file):
    psnrs = AverageMeter()
    ssims = AverageMeter()
    model.eval()
    count = 0
    with torch.no_grad():
        for i, (data,noisy_name) in enumerate(test_loader):
            # test, data convert
            count +=1
            img = data['input'].to(device)
            gt = data['gt'].to(device)
            
            output = model(img)
            output = in_extend_gt(output.clone(), gt)
            
            #save path of test image
            pr_noisy_name = os.path.join(pr_img_path, noisy_name[0])

            # compute metrics in tensor
            psnr, psnr_num = utils.batch_psnr(output, gt)
            ssim, ssim_num = utils.batch_ssim(output, gt)
            psnrs.update(psnr/psnr_num, psnr_num)
            ssims.update(ssim/ssim_num, ssim_num)

            print(noisy_name[0], psnr.item(), ssim)
            metric_file.write("{0:<10}\t{1:<10.8s}\t{2:<10.8s}\n".format(noisy_name[0], str(psnr.item()), str(ssim.item())))

            # images, float-->uint8, gpu-->cpu
            output = torch.clamp(output.cpu()[0], 0, 1).detach().numpy()
            output = np.transpose(output, [1, 2, 0])
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            output = (np.clip(output, 0, 1)*255).astype(np.uint8)

            cv2.imwrite(pr_noisy_name, output)
    print(psnrs.avg, ssims.avg)
    return psnrs.avg, ssims.avg


def in_extend_gt(output, gt):
    _, _, h1, w1 = output.size()
    _, _, h2, w2 = gt.size()
    ex_h = 0
    ex_w = 0
    if h1>h2:
      ex_h = 1
    if w1>w2:
      ex_w = 1
    out = output[:, :, 0:h1-ex_h, 0:w1-ex_w]
    return out
    

if __name__=='__main__':
    main()
