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

from datasets.load_dataset import LoadMSR_dataset_test
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
    parser = argparse.ArgumentParser(description='PyTorch implementation of test')
    parsers = TestArgs()
    args = parsers.initialize(parser)
    parsers.print_args()
    
    ###############################################################################################
    # load netwoork
    print('===> Loading the network ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(device)
    print_model_parm_nums(model)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
  
    if not os.path.exists(args.results_save_linear_path):
        os.mkdir(args.results_save_linear_path) 
    if not os.path.exists(args.results_save_srgb_path):
        os.mkdir(args.results_save_srgb_path) 
    if not os.path.exists(args.metrics_save_path):
        os.mkdir(args.metrics_save_path)
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
    # test data
    test_set = LoadMSR_dataset_test(args.test_datalist)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False, pin_memory=True)

    metrics_img_path = os.path.join(args.metrics_save_path, 'metrics_img.txt')
    metrics_file = open(metrics_img_path, 'w')
    metrics_file.write("{0:<10}\t{1:<10}\t{2:<10}\n".format('img_name', 'psnr', 'ssim'))

    psnr, ssim = test(test_loader, model, args, device, args.results_save_linear_path, args.results_save_srgb_path, metrics_file, \
    args.gammaparams_path, args.colortrans_path)
        
    metrics_file.write("{0:<10.8s}\t{1:<10.8s}\n".format('MSR_test', str(psnr.item()), str(ssim.item())))
    metrics_file.close()

############################################################################################
#
#   functions
#
############################################################################################
def test(test_loader, model, args, device, pr_img_linear_path, pr_img_srgb_path, metrics_file, gammaparams_path, colortrans_path):
    psnrs = AverageMeter()
    ssims = AverageMeter()
    model.eval()
    count = 0
    psnr_srgb = []
    psnr_linear = []

    with torch.no_grad():
        for i, (data,noisy_name) in enumerate(test_loader):
            # test, data convert
            count +=1
            img = data['input'].to(device)
            gt = data['gt'].to(device)
            
            output = model(img)
            output = in_extend_gt(output.clone(), gt)
            
            # save path of test image
            pr_noisy_linear_name = os.path.join(pr_img_linear_path, noisy_name[0])
            pr_noisy_srgb_name = os.path.join(pr_img_srgb_path, noisy_name[0])

            # compute metrics in tensor
            psnr, psnr_num = utils.batch_psnr(output, gt)
            ssim, ssim_num = utils.batch_ssim(output, gt)
            psnrs.update(psnr/psnr_num, psnr_num)
            ssims.update(ssim/ssim_num, ssim_num)

            print(noisy_name[0], psnr.item(), ssim)
            metrics_file.write("{0:<10}\t{1:<10.8s}\t{2:<10.8s}\n".format(noisy_name[0], str(psnr.item()), str(ssim.item())))

            # images, float-->uint8, gpu-->cpu
            gt = torch.clamp(gt.cpu()[0], 0, 1).detach().numpy()
            gt = np.transpose(gt, [1, 2, 0]) 	# permute
            gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
            output = torch.clamp(output.cpu()[0], 0, 1).detach().numpy()
            output = np.transpose(output, [1, 2, 0])
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            gt = gt*255
            output = output*255
            cv2.imwrite(pr_noisy_linear_name, output)
            psnr_linear.append(utils.psnr_cal_np(output/255, gt/255))

            # compute psnr in srgb
            noisy_img = np.swapaxes(np.swapaxes(output, 0,2),1,2)
            gt_img = np.swapaxes(np.swapaxes(gt, 0,2),1,2)
            
            # transform linear to srgb
            srgb_params = utils.init_colortransformation_gamma(gammaparams_path, colortrans_path)
            noisy_img = utils.apply_colortransformation_gamma(np.expand_dims(noisy_img,0), srgb_params)
            gt_img = utils.apply_colortransformation_gamma(np.expand_dims(gt_img,0), srgb_params)

            cv2.imwrite(pr_noisy_srgb_name, np.swapaxes(np.swapaxes(noisy_img[0], 1,2),0,2))
            psnr_srgb.append(utils.psnr_cal_np(noisy_img/255, gt_img/255))
    
    print('psnr_linear', np.mean(np.array(psnr_linear))) 
    print('psnr_srgb', np.mean(np.array(psnr_srgb)))  
    print('psnrs_linear_tensor', psnrs.avg)
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
