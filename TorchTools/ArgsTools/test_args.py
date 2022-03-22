import os
import numpy as np
import shutil

class TestArgs():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        
        # basic args
        parser.add_argument('--gpu_num', default='0', type=str,  help='gpu num')
        parser.add_argument('--results_save_path', default='./test_results_images', type=str,  help='the save path of test results images')
        parser.add_argument('--metrics_save_path', default='./test_results_metrics', type=str,  help='the save path of test results metrics')
        parser.add_argument('--method', default='MGCC', type=str,  help='the name of test method')
        
        # realist raw data args
        parser.add_argument('--gammaparams_path', default='./MSR_srgb_transform/gammaparams.npy', type=str,  help='the path of gammaparams')
        parser.add_argument('--colortrans_path', default='./MSR_srgb_transform/colortrans.npy', type=str,  help='the path of colortrans')
        parser.add_argument('--test_datalist', default='./datasets/test_MSR.txt', type=str,  help='the path of test image list of msr')
        parser.add_argument('--results_save_srgb_path', default='./test_results_images_srgb/', type=str,  help='the save path of test results images in sRGB space')
        parser.add_argument('--results_save_linear_path', default='./test_results_images_linear/', type=str,  help='the save path of test results images in linear space')

        # datasets args
        parser.add_argument('--test_noisy_path', default='', type=str,  help='path to the test noisy image')
        parser.add_argument('--test_gt_path', default='', type=str,  help='path to test gt image')
        parser.add_argument('--sigma', default=0, type=int,  help='noise level GAWN')

        # model args
        parser.add_argument('--pretrained_model', default='', type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--model', default='MGCC', type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--norm_type', default=None, type=str,
                            help='dm_block_type(default: rrdb)')
        parser.add_argument('--block_type', default='rrdb', type=str,
                            help='dm_block_type(default: rrdb)')
        parser.add_argument('--act_type', default='prelu', type=str,
                            help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--bias', action='store_true',
                            help='bias of layer')
        parser.add_argument('--channels', default=64, type=int,
                            help='channels')
        parser.add_argument('--n_resblocks', default=6, type=int,
                            help='number of basic blocks')

        self.args = parser.parse_args()
        return self.args

    def print_args(self):
        # print args
        print("==========================================")
        print("==========       CONFIG      =============")
        print("==========================================")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("\n")



