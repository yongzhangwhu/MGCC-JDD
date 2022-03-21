import sys
sys.path.append("..")
import model.common as common
import torch.nn as nn
import torch
import cv2
from PIL import Image
from PIL import ImageFilter
import numpy as np
from thop import profile
from model.common import DownsamplingShuffle
import model.pac as pac
import argparse

"""
given bayer  -> output noise-free RGB
"""

class NET(nn.Module):
    def __init__(self, opt):
        super(NET, self).__init__()
        
        # parameter
        block_type = opt.block_type
        n_feats = opt.channels
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type

        self.block_num = opt.n_resblocks

        ## architecture

        # head
        self.dm_head_r = nn.Sequential(common.ConvBlock(2, n_feats, act_type=act_type, bias=bias))
        self.dm_head_g = nn.Sequential(common.ConvBlock(3, n_feats, act_type=act_type, bias=bias))
        self.dm_head_b = nn.Sequential(common.ConvBlock(2, n_feats, act_type=act_type, bias=bias))
        
        # channels reconstruction module
        if block_type.lower() == 'rcab':
            self.dm_resblock = nn.ModuleList([CRM(n_feats, feature_block='rcab') for _ in range(self.block_num)])
        elif block_type.lower() == 'rrdb':
            self.dm_resblock = nn.ModuleList([CRM(n_feats, feature_block='rrdb') for _ in range(self.block_num)])                    
        else:
            raise RuntimeError('block_type is not supported')

        # tail for long skip
        self.dm_tail_1_r = nn.Sequential(common.ConvBlock(n_feats, n_feats, 3, bias=True))
        self.dm_tail_1_g = nn.Sequential(common.ConvBlock(n_feats, n_feats, 3, bias=True))
        self.dm_tail_1_b = nn.Sequential(common.ConvBlock(n_feats, n_feats, 3, bias=True))

        # tail for unsampling
        self.dm_tail_2_r = nn.Sequential(common.ConvBlock(n_feats, n_feats, 3, bias=True),\
                common.Upsampler(2, n_feats, norm_type, act_type, bias=bias), common.ConvBlock(n_feats, 1, 3, bias=True))
        self.dm_tail_2_g = nn.Sequential(common.ConvBlock(n_feats, n_feats, 3, bias=True),\
                common.Upsampler(2, n_feats, norm_type, act_type, bias=bias), common.ConvBlock(n_feats, 1, 3, bias=True))
        self.dm_tail_2_b = nn.Sequential(common.ConvBlock(n_feats, n_feats, 3, bias=True),\
                common.Upsampler(2, n_feats, norm_type, act_type, bias=bias), common.ConvBlock(n_feats, 1, 3, bias=True))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    
    def forward(self, x):
 
        r = x[:, 0:1, :, :]
        g = x[:, 1:3, :, :]
        b = x[:, 3:4, :, :]
        n = x[:, 4:5, :, :]

        r_feat_head = self.dm_head_r(torch.cat([r, n], 1))
        g_feat_head = self.dm_head_g(torch.cat([g, n], 1))
        b_feat_head = self.dm_head_b(torch.cat([b, n], 1))
   
        for i in range(self.block_num):
            dm_resblock = self.dm_resblock[i]
            if i == 0:
                r_feat, g_feat, b_feat = dm_resblock(r_feat_head, g_feat_head, b_feat_head)
            else:
                r_feat, g_feat, b_feat = dm_resblock(r_feat, g_feat, b_feat)

        # long skip connection
        r_feat = self.dm_tail_1_r(r_feat) + r_feat_head
        g_feat = self.dm_tail_1_g(g_feat) + g_feat_head
        b_feat = self.dm_tail_1_b(b_feat) + b_feat_head

        # unsampling
        r_out = self.dm_tail_2_r(r_feat)
        g_out = self.dm_tail_2_g(g_feat)
        b_out = self.dm_tail_2_b(b_feat)

        out = torch.cat([r_out, g_out, b_out], 1)
        return out

# Channel reconstruction module
class CRM(nn.Module):
    def __init__(self, n_feats, feature_block='rcab'):
        super(CRM, self).__init__()

        # feature extraction part
        if feature_block.lower() == 'rcab':
            self.dm_resblock = common.RG(n_feats, n_RCAB=20)
        elif feature_block.lower() == 'rrdb':
            self.dm_resblock = common.RRDB(n_feats, n_feats, 3, 1, bias = True, act_type='prelu')                   
        else:
            raise RuntimeError('feature_block is not supported')
        
        # channel guidance part
        self.CGB_r_1 = common.CGB(n_feats)
        self.CGB_r_2 = common.CGB(n_feats)
        self.CGB_g_1 = common.CGB(n_feats)
        self.CGB_g_2 = common.CGB(n_feats)
        self.CGB_b_1 = common.CGB(n_feats)
        self.CGB_b_2 = common.CGB(n_feats)

        self.g_conv_post = nn.Sequential(nn.Conv2d(n_feats*2, n_feats, 3, 1, 1), nn.PReLU())

    def forward(self, r_feats, g_feats, b_feats):
        r = self.dm_resblock(r_feats)
        g = self.dm_resblock(g_feats)
        b = self.dm_resblock(b_feats)
        
        r2 = self.CGB_r_1(r, g)
        r2 = self.CGB_r_2(r2, b)

        b2 = self.CGB_b_1(b, g)
        b2 = self.CGB_b_2(b2, r)

        g2_1 = self.CGB_g_1(g, r2)
        g2_2 = self.CGB_g_2(g, b2)
        g2 = self.g_conv_post(torch.cat([g2_1, g2_2], 1))
            
        return r2 + r_feats, g2 + g_feats, b2 + b_feats
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of DDSR')
    parser.add_argument('--denoise', action='store_false',  help='denoise store_true')
    parser.add_argument('--norm_type', default=None, type=str,
                        help='dm_block_type(default: rrdb)')
    parser.add_argument('--block_type', default='rcab', type=str,
                        help='dm_block_type(default: rrdb)')
    parser.add_argument('--act_type', default='prelu', type=str,
                        help='activation layer {relu, prelu, leakyrelu}')
    parser.add_argument('--bias', action='store_false',
                        help='bias of layer')
    parser.add_argument('--channels', default=64, type=int,
                        help='channels')
    parser.add_argument('--n_resblocks', default=4, type=int,
                            help='number of basic blocks')
    args = parser.parse_args()


    net =NET(args)
    inputs = torch.randn(16, 5, 32, 32)
    out = net(inputs)
    print(out.size())

