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

import utils.binvox_visualization
import utils.binvox_rw
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt

from models.generator import Generator
from models.discriminator import Discriminator

import cv2

from PIL import Image

def test_net(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             generator=None,
             discriminator=None):
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
    if discriminator is None or generator is None:
        generator = Generator(cfg)
        discriminator = Discriminator(cfg)

        if torch.cuda.is_available():
            generator = torch.nn.DataParallel(generator, device_ids=[0, 1]).cuda()
            discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1]).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    # ce_loss = torch.nn.CrossEntropyLoss()
    l1_loss = torch.nn.L1Loss()
    # smooth_l1_loss = torch.nn.SmoothL1Loss()
    # huber_loss = torch.nn.HuberLoss(delta=0.5)

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    generator_losses = utils.network_utils_GAN.AverageMeter()
    discriminator_losses = utils.network_utils_GAN.AverageMeter()

    # Switch models to evaluation mode
    generator.eval()
    discriminator.eval()

    vol_write_idx = 0
    min_loss = 10000000.0
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume, ground_truth_volume_mesh) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)
            # ground_truth_volume_mesh = utils.network_utils.var_or_cuda(ground_truth_volume_mesh)

            # Get random volume excepth matches to batch index
            r = range(0, sample_idx) + range(sample_idx + 1, n_batches)
            fake_idx = random.choice(r)
            fake_volume = ground_truth_volumes[fake_idx]

            # Get data from data loader
            ground_truth_volumes = utils.network_utils_GAN.var_or_cuda(ground_truth_volumes)
            # ground_truth_volumes_mesh = utils.network_utils_GAN.var_or_cuda(ground_truth_volumes_mesh)
            rendering_images = utils.network_utils_GAN.var_or_cuda(rendering_images)

            ground_truth_volumes = ground_truth_volumes.float() / 255.

            # Validate Generator
            fake_images = torch.rand(1, 3, 112, 112, 112)
            gen_volumes = generator(fake_images)
            gen_volumes = gen_volumes.float()
            generator_loss = mse_loss(gen_volumes, ground_truth_volumes)

            # Validate Discriminator

            # real data generator
            real_outputs = generator(rendering_images)
            d_loss_real = bce_loss(real_outputs, ground_truth_volumes)
            d_loss_fake = bce_loss(gen_volumes, fake_volume)

            # discriminator loss
            discriminator_loss = (d_loss_real + d_loss_fake) / 2.

            # Append loss and accuracy to average metrics
            generator_losses.update(generator_loss.item())
            discriminator_losses.update(discriminator_loss.item())

            # Volume Visualization
            '''
            gv = generated_volume.cpu().numpy()
            rendering_views = utils.binvox_visualization.get_volume_views(gv, '/home/jzw/work/pix2vox/output/image/test/gv',
                                                        vol_write_idx)
            np.save('/home/jzw/work/pix2vox/output/voxel/gv/gv_' + str(vol_write_idx).zfill(6) + '.npy', gv)
            vol_write_idx = vol_write_idx + 1
            '''

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(real_outputs, th).float()
                _gt_volume = torch.ge(ground_truth_volume, th).float()
                intersection = torch.sum(torch.ge(_volume.mul(_gt_volume), 1)).float()
                union = torch.sum(torch.ge(_volume.add(_gt_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Append generated volumes to TensorBoard
            if output_dir and sample_idx < 3:
                img_dir = output_dir % 'images'
                # Volume Visualization
                gv_true = gen_volumes.cpu().numpy()

                rendering_views = utils.binvox_visualization.get_volume_views(gv_true, './output/image/test/gv_true',
                                                                              epoch_idx)
                # print(np.shape(rendering_views))
                # rendering_views_im = np.array((rendering_views * 255), dtype=np.uint8)
                # test_writer.add_image('Test Sample#%02d/Volume Reconstructed' % sample_idx, rendering_views_im, epoch_idx)

                gv_false = ground_truth_volume_mesh.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_volume_views(gv_false, './output/image/test/gv_fake',
                                                                              epoch_idx)

                # rendering_views_im = np.array((rendering_views * 255), dtype=np.uint8)
                # test_writer.add_image('Test Sample#%02d/Volume GroundTruth' % sample_idx, rendering_views_im, epoch_idx)
                gtv = ground_truth_volume.cpu().numpy()
                rendering_views = utils.binvox_visualization.get_volume_views(gtv,
                                                                              './output/image/test/gtv',
                                                                              epoch_idx)

            # Print sample loss('IoU = %s' removed)
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s GLoss = %.4f DLoss = %.4f'
                  % (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, generator_loss.item(), discriminator_loss.item()
                     ))

    print('[INFO] %s Test[%d] Loss Mean / GLoss = %.4f DLoss = %.4f'
          % (dt.now(), n_samples, generator_losses.avg, discriminator_losses.avg))

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
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx)
        # test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

    return test_iou[taxonomy_id]['iou'][2] # t = 0.40
    # return min_loss
