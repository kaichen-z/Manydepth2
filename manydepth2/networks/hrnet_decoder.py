from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from .hr_layers import *
from .layers import upsample
import pdb

class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, mobile_encoder=False):
        super(HRDepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()
        # decoder
        self.convs = nn.ModuleDict()
        ksize = [7, 5, 5, 3]
        self.convs["up_x9_0"] = ConvBlock(self.num_ch_dec[1],self.num_ch_dec[0])
        self.convs["up_x9_1"] = ConvBlock(self.num_ch_dec[0],self.num_ch_dec[0])
        self.convs["72"] = Attention_Module_Non_Local(self.num_ch_enc[4], self.num_ch_enc[3] * 2, 256, ksize[0])
        # [12, 144, 6, 20]
        self.convs["36"] = Attention_Module_Non_Local(256, self.num_ch_enc[2] * 3, 128, ksize[1])
        # [12, 256, 12, 40]
        self.convs["18"] = Attention_Module_Non_Local(128, self.num_ch_enc[1] * 3 + 64 , 64, ksize[2])
        # [12, 128, 24, 80]
        self.convs["9"] = Attention_Module_Non_Local(64, 64, 32, ksize[3])
        # [12, 64, 48, 160]
        for i in range(4):
            self.convs["dispConvScale{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        feature144 = input_features[4] # 1 (12, 144, 6, 20) b = 4
        feature72 = input_features[3] # 2 (12, 72, 12, 40) b = 3
        feature36 = input_features[2] # 3 (12, 36, 24, 80) b = 2
        feature18 = input_features[1] # 4 (12, 64, 48, 160) b = 1
        feature64 = input_features[0] # 1 (12, 64, 96, 320) b = 0
        x72 = self.convs["72"](feature144, feature72)
        x36 = self.convs["36"](x72, feature36)
        outputs[("disp",3)] = self.sigmoid(self.convs["dispConvScale3"](x36))
        x18 = self.convs["18"](x36, feature18)
        outputs[("disp",2)] = self.sigmoid(self.convs["dispConvScale2"](x18))
        x9 = self.convs["9"](x18, [feature64])
        outputs[("disp",1)] = self.sigmoid(self.convs["dispConvScale1"](x9))
        x6 = self.convs["up_x9_1"](upsample(self.convs["up_x9_0"](x9)))
        outputs[("disp",0)] = self.sigmoid(self.convs["dispConvScale0"](x6))
        pdb.set_trace()
        return outputs

def sum_params(model):
    s = []
    for p in model.parameters():
        dims = p.size()
        n = p.cpu().data.numpy()
        s.append(np.sum(n))
    return sum(s)