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

        # resolution 32 / Volume
        self.e_layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 32, kernel_size=3),
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
            torch.nn.MaxPool3d(kernel_size=2),
        )
        self.e_layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=3),
            torch.nn.BatchNorm3d(256),
            torch.nn.ELU(),
            torch.nn.MaxPool3d(kernel_size=2),
        )
        self.e_layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 256, kernel_size=3),
            torch.nn.BatchNorm3d(256),
            torch.nn.ELU(),
        )
        self.e_layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 128, kernel_size=3),
            torch.nn.BatchNorm3d(128),
            torch.nn.ELU(),
        )

        # Layer Definition (For 3D)
        self.d_layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1024, 256, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU()
        )
        self.d_layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.d_layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.d_layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.d_layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )


    def forward(self, rendering_images):
        rendering_images = rendering_images.permute(1, 0, 3, 2, 4, 5).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images: # For 32 resolution / Volume
            features = img.squeeze(dim=0)
            # print(features.size()) # torch.Size([1, 3, 112, 112, 112])
            features = self.e_layer1(features)
            # print(features.size()) # torch.Size([1, 32, 110, 110, 110])
            features = self.e_layer2(features)
            # print(features.size()) # torch.Size([1, 64, 54, 54, 54])
            features = self.e_layer3(features)
            # print(features.size()) # torch.Size([1, 128, 26, 26, 26])
            features = self.e_layer4(features)
            # print(features.size()) # torch.Size([1, 256, 12, 12, 12])
            features = self.e_layer5(features)
            # print(features.size()) # torch.Size([1, 256, 10, 10, 10])
            features = self.e_layer6(features)
            # print(features.size()) # torch.Size([1, 128, 8, 8, 8])

            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        gen_volumes = []

        for features in image_features:
            gen_volume = features.view(-1, 1024, 4, 4, 4)
            # print(gen_volume.size())  # torch.Size([batch_size, 128, 8, 8, 8])
            gen_volume = self.d_layer1(gen_volume)
            # print(gen_volume.size())  # torch.Size([batch_size, 64, 16, 16, 16])
            gen_volume = self.d_layer2(gen_volume)
            # print(gen_volume.size())  # torch.Size([batch_size, 32, 32, 32, 32])
            gen_volume = self.d_layer3(gen_volume)
            # print(gen_volume.size())  # torch.Size([batch_size, 32, 32, 32, 32])
            gen_volume = self.d_layer4(gen_volume)
            # print(gen_volume.size())  # # torch.Size([batch_size, 8, 32, 32, 32])
            gen_volume = self.d_layer5(gen_volume)
            # print(gen_volume.size())  # torch.Size([batch_size, 1, 32, 32, 32])

            gen_volumes.append(torch.squeeze(gen_volume, dim=1))

        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        # print(gen_volumes.size())      # torch.Size([batch_size, n_views, 32, 32, 32])
        return gen_volumes
