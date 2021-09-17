# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import gc
import torch
import torchvision.models


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition & only use in 2D image processing
        '''
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        '''

        # For resolution 32 / X-ray
        '''
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )
        '''

        # For resolution 64 / X-ray
        '''
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=5),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=5),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=5),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU()
        )
        '''

        # resolution 32 / Volume
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 8, kernel_size=3),
            torch.nn.BatchNorm3d(8),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 16, kernel_size=3),
            torch.nn.BatchNorm3d(16),
            torch.nn.ELU(),
            torch.nn.MaxPool3d(kernel_size=2),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 32, kernel_size=5),
            torch.nn.BatchNorm3d(32),
            torch.nn.ELU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=3),
            torch.nn.BatchNorm3d(64),
            torch.nn.ELU(),
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ELU(),
            torch.nn.MaxPool3d(kernel_size=3),
        )

        # Don't update params in VGG16 & only use in 2D image processing
        '''
        for param in vgg16_bn.parameters():
            param.requires_grad = False
            '''

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 3, 2, 4, 5).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:

            # For 32 resolution / X-ray
            '''
            features = self.vgg(img.squeeze(dim=0))
            print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            print(features.size())    # torch.Size([batch_size, 512, 26, 26])
            features = self.layer2(features)
            print(features.size())    # torch.Size([batch_size, 512, 8, 8])
            features = self.layer3(features)
            print(features.size())    # torch.Size([batch_size, 256, 8, 8])
            '''

            # For 64 resolution / X-ray
            '''
            features = self.vgg(img.squeeze(dim=0))
            # print(features.size())  # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())  # torch.Size([batch_size, 512, 24, 24])
            features = self.layer2(features)
            # print(features.size())  # torch.Size([batch_size, 512, 20, 20])
            features = self.layer3(features)
            # print(features.size())  # torch.Size([batch_size, 512, 16, 16])
            '''

            gc.collect()
            torch.cuda.empty_cache()
            torch.backends.cudnn.enabled = True

            # For 32 resolution / Volume
            features = img.squeeze(dim=0)
            # print(features.size()) # torch.Size([1, 3, 64, 64, 64])
            features = self.layer1(features)
            # print(features.size()) # torch.Size([1, 8, 62, 62, 62])
            features = self.layer2(features)
            # print(features.size()) # torch.Size([1, 16, 30, 30, 30])
            features = self.layer3(features)
            # print(features.size()) # torch.Size([1, 32, 26, 26, 26])
            features = self.layer4(features)
            # print(features.size()) # torch.Size([1, 64, 24, 24, 24])
            features = self.layer5(features)
            # print(features.size()) # torch.Size([1, 128, 8, 8, 8])

            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 8, 8]) / torch.Size([batch_size, n_views, 512, 16, 16])
        return image_features
