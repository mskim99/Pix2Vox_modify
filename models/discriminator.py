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
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(512 * 4 * 4 * 4, 1),
            torch.nn.Sigmoid(),
        )
        '''
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 512, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.BatchNorm3d(512)
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 1024, kernel_size=1, stride=2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.BatchNorm3d(1024)
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.Conv3d(1024, 1024, kernel_size=1, stride=2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Sigmoid()
        )
        '''
    def forward(self, volume):

        features = volume.view(-1, 1, 64, 64, 64)
        # print(features.size()) # torch.Size([1, 1, 64, 64, 64])
        features = self.layer1(features)
        # print(features.size()) # torch.Size([1, 64, 32, 32, 32])
        features = self.layer2(features)
        # print(features.size()) # torch.Size([1, 128, 16, 16, 16])
        features = self.layer3(features)
        # print(features.size()) # torch.Size([1, 256, 8, 8, 8])
        features = self.layer4(features)
        # print(features.size()) # torch.Size([1, 512, 4, 4, 4])
        features = features.view(-1, 512 * 4 * 4 * 4)
        features = self.layer5(features)
        # print(features.size()) # torch.Size([1, 1])
        '''
        features = self.layer6(features)
        # print(features.size()) # torch.Size([batch_size, 512, 4, 4, 4])
        features = self.layer7(features)
        # print(features.size()) # torch.Size([batch_size, 1024, 2, 2, 2])
        features = self.layer8(features)
        # print(features.size()) # torch.Size([batch_size, 1024, 1, 1, 1])
        features = torch.squeeze(features)
        '''
        return features