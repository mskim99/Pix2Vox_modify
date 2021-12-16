# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import gc
import torch
import torchvision.models


class Generator(torch.nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg

        self.e_layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=0.375),
        )
        self.e_layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=0.375),
        )
        self.e_layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=0.375),
        )
        self.e_layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=0.375),
        )
        self.e_layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(1024),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=0.375),
        )
        ''' 
        self.e_layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(2048),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout3d(p=0.375),
        )    
        self.e_layer7 = torch.nn.Sequential(
            torch.nn.Conv3d(2048, 4096, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )
        '''

        # self.linear = torch.nn.Linear(200, 1024 * 4 * 4 * 4)
        '''
        self.d_layer0_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(4096, 2048, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(2048),
            torch.nn.ReLU(inplace=True),
        )   
        self.d_layer0_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(1024),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout3d(p=0.25),
        )
        '''
        self.d_layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout3d(p=0.25),
        )
        self.d_layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout3d(p=0.25),
        )
        self.d_layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout3d(p=0.25),
        )
        self.d_layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout3d(p=0.25),
        )
        # Resolution 64
        '''
        self.d_layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 1, kernel_size=1, bias=False),
            torch.nn.Sigmoid(),
        )
        '''
        # Resolution 128
        self.d_layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout3d(p=0.25),
        )
        self.d_layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=1, bias=False),
            torch.nn.Sigmoid(),
        )


    def forward(self, images):

        images = images.permute(1, 0, 3, 2, 4, 5).contiguous()
        images = torch.split(images, 1, dim=0)

        features = images[0].squeeze(dim=0)
        # print(features.size()) # torch.Size([1, 3, 128, 128, 128])
        features = self.e_layer1(features)
        # print(features.size()) # torch.Size([1, 64, 64, 64, 64])
        features = self.e_layer2(features)
        # print(features.size()) # torch.Size([1, 128, 32, 32, 32])
        features = self.e_layer3(features)
        # print(features.size()) # torch.Size([1, 256, 16, 16, 16])
        features = self.e_layer4(features)
        # print(features.size()) # torch.Size([1, 512, 8, 8, 8])
        features = self.e_layer5(features)
        # print(features.size()) # torch.Size([1, 1024, 4, 4, 4])
        # features = self.e_layer6(features)
        # print(features.size()) # torch.Size([1, 2048, 2, 2, 2])
        '''
        features = self.e_layer7(features)
        # print(features.size()) # torch.Size([1, 4096, 1, 1, 1])
        '''

        '''
        # gen_volume = self.linear(codes)
        # gen_volume = gen_volume.view(-1, 1024, 4, 4, 4)
        gen_volume = self.d_layer0_1(features)
        # print(gen_volume.size()) # torch.Size([1, 2048, 2, 2, 2])    
        '''
        # gen_volume = self.d_layer0_2(features)
        # print(gen_volume.size()) # torch.Size([1, 1024, 4, 4, 4])
        gen_volume = self.d_layer1(features)
        # print(gen_volume.size()) # torch.Size([1, 512, 8, 8, 8])
        gen_volume = self.d_layer2(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 256, 16, 16, 16])
        gen_volume = self.d_layer3(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 128, 32, 32, 32])
        gen_volume = self.d_layer4(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 64, 64, 64, 64])
        gen_volume = self.d_layer5(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 32, 128, 128, 128])
        gen_volume = self.d_layer6(gen_volume)
        # print(gen_volume.size()) # torch.Size([1, 1, 128, 128, 128])
        gen_volume = torch.squeeze(gen_volume)

        return gen_volume

'''
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1024, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )
'''