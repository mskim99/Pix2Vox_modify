# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os
import json
import random
import torch
import torch.backends.cudnn
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable

from torchvision.utils import save_image

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils_GAN
import utils.loss_function
import utils.extract_amplify_features as uaf

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

# from core.test_GAN import test_net
# from models.encoder_GAN import Encoder
# from models.decoder_GAN import Decoder
from models.generator import Generator
from models.discriminator import Discriminator

import numpy as np
import sys
import math
np.set_printoptions(threshold=sys.maxsize)
import itertools
import cv2

def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.TRAIN, cfg.CONST.N_VIEWS_RENDERING, train_transforms),
                                                    batch_size=cfg.CONST.BATCH_SIZE,
                                                    num_workers=cfg.TRAIN.NUM_WORKER,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset_loader.get_dataset(
        utils.data_loaders.DatasetType.VAL, cfg.CONST.N_VIEWS_RENDERING, val_transforms),
                                                  batch_size=1,
                                                  num_workers=1,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TRAIN_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up networks
    generator = Generator(cfg)
    discriminator = Discriminator(cfg)
    print('[DEBUG] %s Parameters in Generator : %d.' % (dt.now(), utils.network_utils_GAN.count_parameters(generator)))
    print('[DEBUG] %s Parameters in Discriminator: %d.' % (dt.now(), utils.network_utils_GAN.count_parameters(discriminator)))

    # Initialize weights of networks
    generator.apply(utils.network_utils_GAN.init_weights)
    discriminator.apply(utils.network_utils_GAN.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        generator_solver = torch.optim.RMSprop(generator.parameters(),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                          )
        discriminator_solver = torch.optim.RMSprop(discriminator.parameters(),
                                            lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                            )

    elif cfg.TRAIN.POLICY == 'sgd':
        generator_solver = torch.optim.SGD(generator.parameters(),
                                         lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM,
                                         )
        discriminator_solver = torch.optim.SGD(discriminator.parameters(),
                                         lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM,
                                         )
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    generator_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(generator_solver,
                                                                milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    discriminator_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_solver,
                                                                milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)


    if torch.cuda.is_available():
        generator = torch.nn.DataParallel(generator).cuda()
        discriminator = torch.nn.DataParallel(discriminator).cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCEWithLogitsLoss()
    # ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    # smooth_l1_loss = torch.nn.SmoothL1Loss()
    # huber_loss = torch.nn.HuberLoss(delta=0.5)

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_loss = 10000000000
    best_epoch = -1

    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=lambda storage, loc: storage)
        init_epoch = checkpoint['epoch_idx']
        # best_iou = checkpoint['best_iou']
        # best_epoch = checkpoint['best_epoch']

        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        print('[INFO] %s Recover complete. Current epoch #%d.' %
              (dt.now(), init_epoch))

    # Summary writer for TensorBoard
    # output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    # log_dir = output_dir + '/logs'
    # log_dir = output_dir % 'logs'
    ckpt_dir = './Output/logs_GAN/checkpoints'
    train_writer = SummaryWriter('./output/logs_GAN/train')
    val_writer = SummaryWriter('./output/logs_GAN/test')

    # Training loop
    dis_batch = 1
    dis_update = False
    dis_accum = 0
    gen_batch = 1
    gen_accum = 0
    gen_update = False
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils_GAN.AverageMeter()
        data_time = utils.network_utils_GAN.AverageMeter()
        generator_losses = utils.network_utils_GAN.AverageMeter()
        discriminator_losses = utils.network_utils_GAN.AverageMeter()
        dice_losses = utils.network_utils_GAN.AverageMeter()
        L2_losses = utils.network_utils_GAN.AverageMeter()
        L1_losses = utils.network_utils_GAN.AverageMeter()

        # switch models to training mode
        generator.train()
        discriminator.train()

        batch_end_time = time()
        test_iou = dict()
        n_batches = len(train_data_loader)

        # Real Image
        '''
        dis_batch = math.floor((epoch_idx + 6) / 10)
        if dis_batch < 1:
            dis_batch = 1
        elif dis_batch > 6:
            dis_batch = 6
        '''

        # Fake Code
        dis_batch = math.floor((epoch_idx + 19) / 10)
        if dis_batch > 15:
            dis_batch = 15

        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes, ground_truth_volumes_mesh) in enumerate(train_data_loader):
            taxonomy_name = taxonomy_names[0] if isinstance(taxonomy_names[0], str) else taxonomy_names[0].item()
            # Measure data time
            data_time.update(time() - batch_end_time)

            # rendering_images = Variable(torch.Tensor(np.random.normal(0, 1, (200))))

            ground_truth_volumes = ground_truth_volumes.float() / 255.
            rendering_images = rendering_images.float() / 255.

            # Get data from data loader
            ground_truth_volumes = utils.network_utils_GAN.var_or_cuda(ground_truth_volumes)
            rendering_images = utils.network_utils_GAN.var_or_cuda(rendering_images)

            ground_truth_volumes = torch.squeeze(ground_truth_volumes)

            # Train Discriminator
            # Train the discriminator for every n_critic iterations
            dis_update = (batch_idx % dis_batch == 0)
            gen_update = (batch_idx % gen_batch == 0)
            if dis_update:
                discriminator.zero_grad()

                gen_volumes = generator(rendering_images).detach()
                dice_loss = utils.loss_function.dice_loss(gen_volumes, ground_truth_volumes, 0.29)
                L2_loss = mse_loss(gen_volumes, ground_truth_volumes)
                L1_loss = l1_loss(gen_volumes, ground_truth_volumes)
                discriminator_loss = - torch.mean(discriminator(ground_truth_volumes)) \
                                     + torch.mean(discriminator(gen_volumes))
                discriminator_loss = discriminator_loss + (L2_loss + dice_loss) / 2.

                if gen_update:
                    discriminator_loss.backward(retain_graph=True)
                else:
                    discriminator_loss.backward()
                discriminator_solver.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # Train Generator
            # Train the generator every n_critic iterations
            if gen_update:
                generator_solver.zero_grad()

                gen_volumes = generator(rendering_images)
                dice_loss = utils.loss_function.dice_loss(gen_volumes, ground_truth_volumes, 0.29)
                L2_loss = mse_loss(gen_volumes, ground_truth_volumes)
                L1_loss = l1_loss(gen_volumes, ground_truth_volumes)
                generator_loss = -torch.mean(discriminator(gen_volumes))
                generator_loss = generator_loss + (L2_loss + dice_loss) / 2.

                generator_loss.backward()
                generator_solver.step()


            generator_loss_value = generator_loss.item()
            discriminator_loss_value = discriminator_loss.item()

            # Append loss to average metrics
            generator_losses.update(generator_loss_value)
            discriminator_losses.update(discriminator_loss_value)
            dice_losses.update(dice_loss.item())
            L2_losses.update(L2_loss.item())
            L1_losses.update(L1_loss.item())

            # Fake Code
            if discriminator_loss_value > 0.05:
                dis_batch = 1
            elif discriminator_loss_value < -0.7:
                dis_batch = math.floor((epoch_idx + 19) / 10)
                if dis_batch > 10:
                    dis_batch = 10

            # Real Image
            '''
            if discriminator_loss_value > -0.05:
                dis_batch = 1
            elif discriminator_loss_value < -0.8:
                dis_batch = math.floor((epoch_idx + 6) / 10)
                if dis_batch < 1:
                    dis_batch = 1
                elif dis_batch > 6:
                    dis_batch = 6
                    '''
            '''
            # Renew discriminator / generator Interval
            if dis_update:
                if discriminator_loss_value > -0.8:
                    dis_batch = 1
                    dis_accum = 0
                if (discriminator_loss_value <= -0.8) & (discriminator_loss_value > -0.9):
                    dis_batch = 3 + 2 * dis_accum
                    dis_accum = dis_accum + 1
                if discriminator_loss_value < -0.9:
                    dis_batch = 3 + 3 * dis_accum
                    dis_accum = dis_accum + 1

            # Renew discriminator / generator Interval
            if gen_update:
                if generator_loss_value > -0.8:
                    gen_batch = 1
                    gen_accum = 0
                if (generator_loss_value <= -0.8) & (generator_loss_value > -0.9):
                    gen_batch = 3 + 2 * gen_accum
                    gen_accum = gen_accum + 1
                if generator_loss_value < -0.9:
                    gen_batch = 5 + 3 * gen_accum
                    gen_accum = gen_accum + 1

            if (not dis_update) & (not gen_update):
                dis_batch = 1
                dis_accum = 0
                gen_batch = 1
                gen_accum = 0
            '''

            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('Generator/BatchLoss', generator_loss.item(), n_itr)
            # train_writer.add_scalar('Generator_fake/BatchLoss', generator_loss_fake.item(), n_itr)
            train_writer.add_scalar('Discriminator/BatchLoss', discriminator_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            print(
                '[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) GLoss = %.6f DLoss = %.6f DiceLoss = %.6f L2Loss = %.6f L1Loss = %.6f'
                % (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val,
                   data_time.val, generator_loss.item(), discriminator_loss.item(), dice_loss.item(), L2_loss.item(), L1_loss.item()))

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(gen_volumes, th).float()
                _gt_volume = torch.ge(ground_truth_volumes, th).float()
                intersection = torch.sum(torch.ge(_volume.mul(_gt_volume), 1)).float()
                union = torch.sum(torch.ge(_volume.add(_gt_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            # IoU per taxonomy
            if taxonomy_name not in test_iou:
                test_iou[taxonomy_name] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_name]['n_samples'] += 1
            test_iou[taxonomy_name]['iou'].append(sample_iou)

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('Generator/EpochLoss', generator_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Discriminator/EpochLoss', discriminator_losses.avg, epoch_idx + 1)

        # Adjust learning rate
        generator_lr_scheduler.step()
        discriminator_lr_scheduler.step()

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) GLoss = %.8f DLoss = %.8f DiceLoss = %.8f L2Loss = %.6f L1Loss = %.6f' %
              (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time,
               generator_losses.avg, discriminator_losses.avg, dice_losses.avg, L2_losses.avg, L1_losses.avg))

        gv = gen_volumes.detach().cpu().numpy()
        np.save('./output/voxel_GAN/gv/gv_' + str(epoch_idx).zfill(6) + '.npy', gv)

        gtv = ground_truth_volumes.cpu().numpy()
        np.save('./output/voxel_GAN/gtv/gtv_' + str(epoch_idx).zfill(6) + '.npy', gtv)

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' %
                  (dt.now(), epoch_idx + 2, cfg.TRAIN.NUM_EPOCHES, n_views_rendering))

        # Validate the training models
        # iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, discriminator)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils_GAN.save_checkpoints(cfg, './output/logs_GAN/checkpoints/ckpt-epoch-%04d.pth' % (epoch_idx + 1),
                                                 epoch_idx + 1, generator, generator_solver,
                                                 discriminator, discriminator_solver)

        # Output testing results
        mean_iou = []
        for taxonomy_id in test_iou:
            test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
            mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
        mean_iou = np.sum(mean_iou, axis=0) / cfg.TRAIN.NUM_EPOCHES

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

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
