# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Refiner(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer_encoder1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer_encoder2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer_encoder3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer_encoder4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer_fc1 = torch.nn.Sequential(
            # torch.nn.Linear(8192, 2048),
            torch.nn.Linear(16384, 2048),
            torch.nn.ReLU()
        )
        self.layer_fc2 = torch.nn.Sequential(
            # torch.nn.Linear(2048, 8192),
            torch.nn.Linear(2048, 16384),
            torch.nn.ReLU()
        )
        self.layer_decoder0 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer_decoder1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer_decoder2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer_decoder3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, coarse_volumes):

        volumes_64_l = coarse_volumes.view((-1, 1, 64, 64, 64))
        # print(volumes_64_l.size())       # torch.Size([batch_size, 1, 64, 64, 64])
        volumes_32_l = self.layer_encoder1(volumes_64_l)
        # print(volumes_32_l.size())       # torch.Size([batch_size, 32, 32, 32, 32])
        volumes_16_l = self.layer_encoder2(volumes_32_l)
        # print(volumes_16_l.size())        # torch.Size([batch_size, 64, 16, 16, 16])
        volumes_8_l = self.layer_encoder3(volumes_16_l)
        # print(volumes_8_l.size())        # torch.Size([batch_size, 128, 8, 8, 8])
        volumes_4_l = self.layer_encoder4(volumes_8_l)
        # print(volumes_4_l.size())        # torch.Size([batch_size, 256, 4, 4, 4])
        flatten_features = self.layer_fc1(volumes_4_l.view(-1, 16384))
        # print(flatten_features.size())   # torch.Size([batch_size, 2048])
        flatten_features = self.layer_fc2(flatten_features)
        # print(flatten_features.size())   # torch.Size([batch_size, 16384])
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 256, 4, 4, 4)
        # print(volumes_4_r.size())        # torch.Size([batch_size, 256, 4, 4, 4])
        volumes_8_r = volumes_8_l + self.layer_decoder0(volumes_4_r)
        # print(volumes_8_r.size())        # torch.Size([batch_size, 128, 8, 8, 8])
        volumes_16_r = volumes_16_l + self.layer_decoder1(volumes_8_r)
        # print(volumes_16_r.size())       # torch.Size([batch_size, 64, 16, 16, 16])
        volumes_32_r = (volumes_32_l + self.layer_decoder2(volumes_16_r))
        # print(volumes_32_r.size())       # torch.Size([batch_size, 32, 32, 32, 32])
        volumes_64_r = (volumes_64_l + self.layer_decoder3(volumes_32_r)) * 0.5
        # print(volumes_64_r.size())       # torch.Size([batch_size, 1, 64, 64, 64])

        return volumes_64_r.view((-1, 64, 64, 64))

        # For 32 resolution
        '''
        volumes_32_l = coarse_volumes.view((-1, 1, 32, 32, 32))
        # print(volumes_32_l.size())       # torch.Size([batch_size, 1, 32, 32, 32])
        volumes_16_l = self.layer_encoder1(volumes_32_l)
        # print(volumes_16_l.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_8_l = self.layer_encoder2(volumes_16_l)
        # print(volumes_8_l.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_4_l = self.layer_encoder3(volumes_8_l)
        # print(volumes_4_l.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        flatten_features = self.layer_fc1(volumes_4_l.view(-1, 8192))
        # print(flatten_features.size())   # torch.Size([batch_size, 2048])
        flatten_features = self.layer_fc2(flatten_features)
        # print(flatten_features.size())   # torch.Size([batch_size, 8192])
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 4, 4, 4)
        # print(volumes_4_r.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        volumes_8_r = volumes_8_l + self.layer_decoder1(volumes_4_r)
        # print(volumes_8_r.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_16_r = volumes_16_l + self.layer_decoder2(volumes_8_r)
        # print(volumes_16_r.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_32_r = (volumes_32_l + self.layer_decoder3(volumes_16_r)) * 0.5
        # print(volumes_32_r.size())       # torch.Size([batch_size, 1, 32, 32, 32])        
        

        return volumes_32_r.view((-1, 32, 32, 32))
        '''
