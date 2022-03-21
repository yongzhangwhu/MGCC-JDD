import glob
import os
import re
import os
import sys
import datetime
import logging
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


'''
# ===============================
# logger_info
# set log arguments 
# ===============================
'''
def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exists!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


'''
# ===============================
# print to file and std_out simultaneously
# ===============================
'''
class logger_print(object):
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  # write the message

    def flush(self):
        pass

'''
# set log
'''
def set_log(model_name, log_path):
    logger_name = model_name
    logger_info(logger_name, log_path=log_path)
    logger = logging.getLogger(logger_name)
    return logger

'''
# --------------------------------------------
# findLastCheckpoint(save_dir)
# finding the last model and loading to net
# --------------------------------------------
'''
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


'''
# --------------------------------------------
# batch_ssim(img, imclean)
# compute ssim of a batch
# --------------------------------------------
'''
def batch_ssim(img, imclean):
    r"""
    Computes the SSIM along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
    """
    img = img.permute(0,2,3,1).cpu().numpy().astype(np.float32)
    batch_num = img.shape[0]
    imgclean = imclean.permute(0,2,3,1).cpu().numpy().astype(np.float32)
    batch_ssim = 0
    for i in range(batch_num):
        batch_ssim += structural_similarity(imgclean[i, :, :, :], img[i, :, :, :], multichannel=True)
    return batch_ssim, batch_num


'''
# --------------------------------------------
# batch_psnr(img, imclean)
# compute psnr of a batch
# --------------------------------------------
'''
def batch_psnr(img, imclean):
    r"""
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
    """
    batch_num = img.size()[0]
    batch_psnr = 0
    for i in range(batch_num):
        batch_psnr += psnr_cal(img[i,:,:,:], imclean[i,:,:,:])
    return batch_psnr, batch_num


'''
# --------------------------------------------
# psnr_cal(img, gt)
# compute psnr of a single image
# --------------------------------------------
'''
def psnr_cal(img, gt):
    mse = torch.mean((img - gt) ** 2)
    return 20. * torch.log10(1. / torch.sqrt(mse))

def psnr_cal_np(img, gt):
    mse = np.mean((img - gt) ** 2)
    return 20. * np.log10(1. / np.sqrt(mse))


################################################################
# linear image--->sRGB image for MSR dataset
#
################################################################
#load parameters for the gamma transformation
#the parameters are particular for the given data, and taken from
#the MSR demosaicing dataset
def init_colortransformation_gamma(gammaparams_path, colortrans_path):
    gammaparams = np.load(gammaparams_path).astype('float32')
    colortrans_mtx = np.load(colortrans_path).astype('float32')
    colortrans_mtx = np.expand_dims(np.expand_dims(colortrans_mtx,0),0)

    param_dict = {
        'UINT8' :  255.0,
        'UINT16' : 65535.0,
        'corr_const' : 15.0,
        'gammaparams' : gammaparams,
        'colortrans_mtx' : colortrans_mtx,
    }

    return param_dict

#compute the gamma function
#we fitted a function according to the given gamma mapping in the
#Microsoft demosaicing data set
def _f_gamma(img, param_dict):
    params = param_dict['gammaparams']
    UINT8 = param_dict['UINT8']
    UINT16 = param_dict['UINT16']

    return UINT8*(((1 + params[0]) * \
        np.power(UINT16*(img/UINT8), 1.0/params[1]) - \
        params[0] +
        params[2]*(UINT16*(img/UINT8)))/UINT16)

#apply the color transformation matrix
def _f_color_t(img, param_dict):
    return  np.tensordot(param_dict['colortrans_mtx'], img, axes=([1,2],[0,1]))

#apply the black level correction constant
def _f_corr(img, param_dict):
    return img - param_dict['UINT8'] * \
         (param_dict['corr_const']/param_dict['UINT16'])

#wrapper for the conversion from linear to sRGB space with given parameters
def apply_colortransformation_gamma(img, param_dict):
    img = _f_color_t(img, param_dict)
    img = np.where( img > 0.0, _f_gamma(img, param_dict), img )
    img = _f_corr(img, param_dict)

    return img

if __name__ == '__main__':
    x1 = torch.randn(12, 4, 256, 256)
    x2 = torch.randn(12, 4, 256, 256)
    batch_msgs, batch_num = batch_msgs(x1, x2)
    print(batch_msgs, batch_num )