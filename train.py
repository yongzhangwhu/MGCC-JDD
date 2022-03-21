import os
import argparse
import shutil
import importlib
import numpy as np
from tqdm import tqdm

import torch
import time
from torch.utils.data import DataLoader

from datasets.load_dataset import Load_Synthetic_train, Load_Synthetic_valid
from model.common import print_model_parm_nums
from TorchTools.ArgsTools.base_args import BaseArgs
from TorchTools.LogTools.logger_tensorboard import Tf_Logger
from TorchTools.LossTools.loss import C_Loss, VGGLoss
from TorchTools.LossTools.metrics import PSNR, AverageMeter
import utils 


def main():
    ###############################################################################################
    # args parse
    parser = argparse.ArgumentParser(description='PyTorch implementation of MGCC for train')
    parsers = BaseArgs()
    args = parsers.initialize(parser)
    parsers.print_args()
    ###############################################################################################
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    print('===> Creating dataloader...')
    train_set = Load_Synthetic_train(args.train_list, args.patch_size, args.max_noise, args.min_noise)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True)

    valid_set = Load_Synthetic_valid(args.valid_list, args.patch_size, args.max_noise, args.min_noise)
    valid_loader = DataLoader(dataset=valid_set, num_workers=4, batch_size=args.batch_size,
                              shuffle=False, pin_memory=True)

    ###############################################################################################
    print('===> Loading the network ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(device)
    print_model_parm_nums(model)

    ###############################################################################################
    # load checkpoints
    best_psnr = 0
    cur_psnr = 0
    cur_ssim = 0
    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            print("=====> loading checkpoint '{}'".format(args.pretrained_model))
            checkpoint = torch.load(args.pretrained_model)
            best_psnr = checkpoint['best_psnr']
            #args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("The pretrained_model is at checkpoint {}, and it's best loss is {}."
                  .format(checkpoint['iter'], best_psnr))
        else:
            print("=====> no checkpoint found at '{}'".format(args.pretrained_model))

    ###############################################################################################
    # optimize
    log = utils.set_log(args.model, args.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min =1e-6, verbose=True)
    
    ###############################################################################################
    # train + valid
    logger = Tf_Logger(args.logdir)
    losses = AverageMeter()
    current_iter = args.start_iters
    
    print('---------- Start training -------------')
    for epoch in range(args.total_epochs):
        ###########################################
        # train
        for i, data in enumerate(train_loader):
            current_iter += 1
            if current_iter > args.total_iters:
                break

            model.train()
            # train, data convert
            img = data['input'].to(device)
            gt = data['gt'].to(device)
            
            # zero parameters
            optimizer.zero_grad()

            # net
            output = model(img)

            # forward+backward+optim#
            loss = torch.mean(torch.abs(output-gt)).cuda()

            # optim
            loss.backward()
            optimizer.step()
            
            lr_update = optimizer.param_groups[0]['lr']
            losses.update(loss.item(), gt.size(0))

            if current_iter % args.print_freq == 0:
                log.info('Iter:{0}       '
                      'lr:{1}       '
                      'Loss {loss.val: .4f} ({loss.avg: .4f})\t'
                      .format(
                       current_iter, lr_update, loss=losses))
    
            # valid
            scheduler.step(current_iter / args.valid_freq)
            if current_iter % args.valid_freq == 0:
                model.eval()
                cur_psnr, cur_ssim = valid(valid_loader, model, current_iter, args, device, logger, log)
                #scheduler.step()
            
            # log
            info = {
                'loss': loss,
                'psnr': cur_psnr,
                'ssim': cur_ssim
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, current_iter)

            #######################################################################
            # save checkpoints
            if current_iter % args.save_freq == 0:
                is_best = (cur_psnr > best_psnr)
                best_psnr = max(cur_psnr, best_psnr)
                model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                save_checkpoint({
                    'iter': current_iter,
                    'state_dict': model_cpu,
                    'best_psnr': best_psnr,
                    'lr': lr_update
                }, is_best, args=args)

    print('Saving the final model.')

############################################################################################
#
#   functions
#
############################################################################################
def valid(valid_loader, model, iter, args, device, logger, log):
    psnrs = AverageMeter()
    ssims = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # valid, data convert
            img = data['input'].to(device)
            gt = data['gt'].to(device)
            output = model(img)

            # psnr
            psnr, psnr_num = utils.batch_psnr(output, gt)
            ssim, ssim_num = utils.batch_ssim(output, gt)
            
            if psnr < 1000:
                psnrs.update(psnr/psnr_num, psnr_num)
                ssims.update(ssim/ssim_num, ssim_num)

            log.info('Iter: [{0}][{1}/{2}]\t''Valid PSNR: ({psnrs.val: .4f}, {psnrs.avg: .4f})\t Valid SSIM: ({ssims.val: .4f}, {ssims.avg: .4f})\t'.format(
                   iter, i, len(valid_loader), psnrs=psnrs, ssims = ssims))

    # show images
    img = img[0:1, 0:4]
    img = torch.clamp(img.cpu()[0, 0], 0, 1).detach().numpy()
    gt = torch.clamp(gt.cpu()[0], 0, 1).detach().numpy()
    gt = np.transpose(gt, [1, 2, 0]) 	# permute
    output = torch.clamp(output.cpu()[0], 0, 1).detach().numpy()
    output = np.transpose(output, [1, 2, 0])

    vis = [img, output, gt]
    logger.image_summary(args.post, vis, iter)
    return psnrs.avg, ssims.avg

#  save checkpoint
def save_checkpoint(state, is_best, args):
    filename = '{}/{}_checkpoint_{}k.path'.format(args.save_path, args.post, state['iter']/1000)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}_model_best.path'.format(args.save_path, args.post))


if __name__=='__main__':
    main()
