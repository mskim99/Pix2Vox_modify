# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import random

import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
from torch.autograd import Variable

import utils.binvox_visualization
import utils.binvox_rw
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt

from models.generator import Generator

import cv2
import itertools
from PIL import Image

import utils.extract_amplify_features as uaf

def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             generator=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=1,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Set up networks
    if generator is None:
        generator = Generator(cfg)

        if torch.cuda.is_available():
            generator = torch.nn.DataParallel(generator).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        generator.load_state_dict(checkpoint['generator_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()
    # ce_loss = torch.nn.CrossEntropyLoss()
    l1_loss = torch.nn.L1Loss()
    # smooth_l1_loss = torch.nn.SmoothL1Loss()
    # huber_loss = torch.nn.HuberLoss(delta=0.5)

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    L1_losses = utils.network_utils_GAN.AverageMeter()
    dice_losses = utils.network_utils_GAN.AverageMeter()
    SSIM_losses = utils.network_utils_GAN.AverageMeter()
    iou_losses = utils.network_utils_GAN.AverageMeter()

    # Switch models to evaluation mode
    generator.eval()

    n_batches = len(test_data_loader)
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volumes, ground_truth_volume_mesh) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)

            ground_truth_volumes = ground_truth_volumes.float() / 255.
            rendering_images = rendering_images / 255.

            ground_truth_volumes = torch.squeeze(ground_truth_volumes)

            # Train Generator
            gen_volumes = generator(rendering_images)
            dice_loss = utils.loss_function.dice_loss(gen_volumes, ground_truth_volumes, 0.29)
            L2_loss = mse_loss(gen_volumes, ground_truth_volumes)
            L1_loss = l1_loss(gen_volumes, ground_truth_volumes)
            SSIM_loss = utils.loss_function.ssim_loss_volume(gen_volumes, ground_truth_volumes)
            SSIM_loss = 0.5 - SSIM_loss

            sample_iou = []
            for th in [.3, .4, .5]:
                _volume = torch.ge(gen_volumes, th).float()
                _gt_volume = torch.ge(ground_truth_volumes, th).float()
                intersection = torch.sum(torch.ge(_volume.mul(_gt_volume), 1)).float()
                union = torch.sum(torch.ge(_volume.add(_gt_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            iou_loss = sum(sample_iou) / len(sample_iou)
            iou_loss = 0.5 - iou_loss

            # Append loss to average metrics
            L1_losses.update(L1_loss.item())
            dice_losses.update(dice_loss.item())
            SSIM_losses.update(SSIM_loss.item())
            iou_losses.update(iou_loss)

            # Volume Visualization
            '''
            gv = gen_volumes.cpu().numpy()
            np.save('/home/jzw/work/pix2vox/output/voxel_test/gv/gv_' + str(sample_idx).zfill(6) + '.npy', gv)
            gtv = ground_truth_volumes.cpu().numpy()
            np.save('/home/jzw/work/pix2vox/output/voxel_test/gtv/gtv_' + str(sample_idx).zfill(6) + '.npy', gtv)
            '''

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(gen_volumes, th).float()
                _gt_volume = torch.ge(ground_truth_volumes, th).float()
                intersection = torch.sum(torch.ge(_volume.mul(_gt_volume), 1)).float()
                union = torch.sum(torch.ge(_volume.add(_gt_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s IoULoss = %.6f SSIMLoss = %.6f L1Loss = %.6f'
                  % (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, iou_loss, SSIM_loss.item(), L1_loss.item()))

    print('[INFO] %s Test[%d] Loss Mean / IoULoss = %.6f SSIMLoss = %.6f L1Loss = %.6f'
          % (dt.now(), n_samples, iou_losses.avg, SSIM_losses.avg, L1_losses.avg))

    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('Generator/DiceLoss', dice_losses.avg, epoch_idx)
        test_writer.add_scalar('Generator/SSIMLoss', SSIM_losses.avg, epoch_idx)
        test_writer.add_scalar('Generator/L1Loss', L1_losses.avg, epoch_idx)

    return test_iou[taxonomy_id]['iou'][2] # t = 0.40
    # return min_loss
