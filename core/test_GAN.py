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

import joblib

def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             generator=None,
             volume_scaler=None,
             image_scaler=None):
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

    if volume_scaler is None or image_scaler is None:
        volume_scaler = joblib.load('./output/logs/checkpoints/volume_scaler.pkl')
        image_scaler = joblib.load('./output/logs/checkpoints/image_scaler.pkl')

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
    L1_losses_thres = utils.network_utils_GAN.AverageMeter()
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

            if volume_scaler is not None and image_scaler is not None:
                ground_truth_volumes = volume_scaler.transform(ground_truth_volumes.reshape(-1, ground_truth_volumes.shape[-1])).reshape(ground_truth_volumes.shape)
                rendering_images = image_scaler.transform(rendering_images.reshape(-1, rendering_images.shape[-1])).reshape(rendering_images.shape)

                ground_truth_volumes = torch.from_numpy(ground_truth_volumes).type(torch.FloatTensor)
                rendering_images = torch.from_numpy(rendering_images).type(torch.FloatTensor)
            else:
                ground_truth_volumes = ground_truth_volumes.float() / 255.
                rendering_images = rendering_images / 255.

            '''
            rendering_images = image_scaler.transform(rendering_images.reshape(-1, rendering_images.shape[-1])).reshape(rendering_images.shape)
            rendering_images = torch.from_numpy(rendering_images).type(torch.FloatTensor)
            ground_truth_volumes = ground_truth_volumes.float() / 255.
            '''
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)

            ground_truth_volumes = torch.squeeze(ground_truth_volumes)

            # Train Generator
            gen_volumes = generator(rendering_images)
            dice_loss = utils.loss_function.dice_loss(gen_volumes, ground_truth_volumes, 0.29)
            L2_loss = mse_loss(gen_volumes, ground_truth_volumes)
            L1_loss = l1_loss(gen_volumes, ground_truth_volumes)
            SSIM_loss = utils.loss_function.ssim_loss_volume(gen_volumes, ground_truth_volumes)
            SSIM_loss = 0.5 - SSIM_loss

            # gen_volumes_thres_pos = torch.ge(gen_volumes, 0.39)
            gt_volumes_thres_pos = torch.ge(ground_truth_volumes, 0.39)
            # gen_volumes_thres = gen_volumes * gen_volumes_thres_pos
            gt_volumes_thres = ground_truth_volumes * gt_volumes_thres_pos
            L1_loss_thres = l1_loss(gen_volumes, gt_volumes_thres)

            sample_iou = []
            for th in [.2, .3, .4, .5]:
                _volume = torch.ge(gen_volumes, th).float()
                _gt_volume = torch.ge(ground_truth_volumes, th).float()
                intersection = torch.sum(torch.ge(_volume.mul(_gt_volume), 1)).float()
                union = torch.sum(torch.ge(_volume.add(_gt_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            iou_loss = sum(sample_iou) / len(sample_iou)
            iou_loss = 0.5 - iou_loss

            sample_iou.clear()

            # Append loss to average metrics
            L1_losses.update(L1_loss.item())
            L1_losses_thres.update(L1_loss_thres.item())
            dice_losses.update(dice_loss.item())
            SSIM_losses.update(SSIM_loss.item())
            iou_losses.update(iou_loss)

            # Volume Visualization
            '''
            gv = gen_volumes.cpu().numpy()
            np.save('./output/voxel2/gv/gv_' + str(sample_idx).zfill(6) + '.npy', gv)

            gtv = ground_truth_volumes.cpu().numpy()
            np.save('./output/voxel2/gtv/gtv_' + str(sample_idx).zfill(6) + '.npy', gtv)
            '''

            # IoU per sample
            sample_iou = []
            sample_accuracy = []
            sample_precision = []
            sample_recall = []
            sample_f1_score = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(gen_volumes, th).float()
                _gt_volume = torch.ge(ground_truth_volumes, th).float()

                volume_num = torch.sum(_volume).float()
                gt_volume_num = torch.sum(_gt_volume).float()
                total_voxels = float(128 * 128 * 128)

                intersection = torch.sum(torch.ge(_volume.mul(_gt_volume), 1)).float()
                union = torch.sum(torch.ge(_volume.add(_gt_volume), 1)).float()

                iou = intersection / union

                TP = intersection / total_voxels
                TN = 1.0 - (union / total_voxels)
                FP = (volume_num - intersection) / total_voxels
                FN = (gt_volume_num - intersection) / total_voxels

                # print(str(TP) + ' ' + str(FP) + ' ' + str(TN) + ' ' + str(FN))

                precision = TP / (TP + FP)
                recall = TP / (TP + FN)

                sample_iou.append(iou.item())
                sample_accuracy.append(((TP + TN) / (TP + TN + FP + FN)).item())
                sample_precision.append(precision.item())
                sample_recall.append(recall.item())
                sample_f1_score.append(((2. * precision * recall) / (precision + recall)).item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': [], 'accuracy' : [], 'precision' : [], 'recall' : [], 'f1-score' : []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)
            test_iou[taxonomy_id]['accuracy'].append(sample_accuracy)
            test_iou[taxonomy_id]['precision'].append(sample_precision)
            test_iou[taxonomy_id]['recall'].append(sample_recall)
            test_iou[taxonomy_id]['f1-score'].append(sample_f1_score)


            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s IoULoss = %.6f SSIMLoss = %.6f L1Loss = %.6f L1Loss_thres = %.6f'
                  % (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, iou_loss, SSIM_loss.item(), L1_loss.item(), L1_loss_thres.item()))

    print('[INFO] %s Test[%d] Loss Mean / IoULoss = %.6f SSIMLoss = %.6f L1Loss = %.6f L1Loss_thres = %.6f'
          % (dt.now(), n_samples, iou_losses.avg, SSIM_losses.avg, L1_losses.avg, L1_losses_thres.avg))

    # print(test_iou[taxonomy_id]['recall'])

    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        test_iou[taxonomy_id]['accuracy'] = np.mean(test_iou[taxonomy_id]['accuracy'], axis=0)
        test_iou[taxonomy_id]['precision'] = np.mean(test_iou[taxonomy_id]['precision'], axis=0)
        test_iou[taxonomy_id]['recall'] = np.mean(test_iou[taxonomy_id]['recall'], axis=0)
        test_iou[taxonomy_id]['f1-score'] = np.mean(test_iou[taxonomy_id]['f1-score'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples


    eval_factors = ['iou', 'accuracy', 'precision', 'recall', 'f1-score']
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
        for eval_factor in eval_factors:
            print('%s' % eval_factor.ljust(8), end='\t')
            print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
            if 'baseline' in taxonomies[taxonomy_id]:
                print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
            else:
                print('N/a', end='\t\t')

            for ti in test_iou[taxonomy_id][eval_factor]:
                print('%.4f' % ti, end='\t')
            print('')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('Generator/DiceLoss', dice_losses.avg, epoch_idx)
        test_writer.add_scalar('Generator/SSIMLoss', SSIM_losses.avg, epoch_idx)
        test_writer.add_scalar('Generator/L1Loss', L1_losses.avg, epoch_idx)

    return test_iou[taxonomy_id]['iou'][2] # t = 0.40
    # return min_loss
