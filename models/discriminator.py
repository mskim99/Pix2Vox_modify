# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import gc
import torch
import torchvision.models


class Discriminator(torch.nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.cfg = cfg

        # resolution 32 / Volume
        self.e_layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=3),
            torch.nn.BatchNorm3d(32),
            torch.nn.ELU(),
        )
        self.e_layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=3),
            torch.nn.BatchNorm3d(64),
            torch.nn.ELU(),
            torch.nn.MaxPool3d(kernel_size=2),
        )
        self.e_layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=3),
            torch.nn.BatchNorm3d(128),
            torch.nn.ELU(),
        )
        self.e_layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=3),
            torch.nn.BatchNorm3d(256),
            torch.nn.ELU(),
        )
        self.e_layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=3),
            torch.nn.BatchNorm3d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool3d(kernel_size=2),
        )

        # Layer Definition (For 3D)
        '''
        self.d_layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.d_layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.d_layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.d_layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )
        '''

    def forward(self, volume):
        # print(volume.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        features = volume.view((-1, 1, 32, 32, 32))

        # For 32 resolution / Volume
        # print(features.size()) # torch.Size([1, 1, 32, 32, 32])
        features = self.e_layer1(features)
        # print(features.size()) # torch.Size([1, 32, 30, 30, 30])
        features = self.e_layer2(features)
        # print(features.size()) # torch.Size([1, 64, 14, 14, 14])
        features = self.e_layer3(features)
        # print(features.size()) # torch.Size([1, 128, 12, 12, 12])
        features = self.e_layer4(features)
        # print(features.size()) # torch.Size([1, 256, 10, 10, 10])
        features = self.e_layer5(features)
        # print(features.size()) # torch.Size([1, 512, 4, 4, 4])

        '''
        features = self.d_layer1(features)
        # print(gen_volume.size())  # torch.Size([1, 128, 8, 8, 8])
        features = self.d_layer2(features)
        # print(gen_volume.size())  # torch.Size([1, 32, 16, 16, 16])
        features = self.d_layer3(features)
        # print(gen_volume.size())  # torch.Size([1, 8, 32, 32, 32])
        features = self.d_layer4(features)
        # print(gen_volume.size())  # # torch.Size([1, 1, 32, 32, 32])
        '''

        features = torch.flatten(features)
        # print(features.size())  # torch.Size([batch_size, n_views, 256, 8, 8]) / torch.Size([batch_size, n_views, 512, 16, 16])
        return features
