# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os
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
        generator_solver = torch.optim.Adam(generator.parameters(),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS,
                                          weight_decay=0.2
                                          )
        discriminator_solver = torch.optim.Adam(discriminator.parameters(),
                                            lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                            betas=cfg.TRAIN.BETAS,
                                            weight_decay=0.2
                                            )

    elif cfg.TRAIN.POLICY == 'sgd':
        generator_solver = torch.optim.SGD(generator.parameters(),
                                         lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM,
                                         weight_decay=0.2
                                         )
        discriminator_solver = torch.optim.SGD(discriminator.parameters(),
                                         lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM
                                         , weight_decay=0.2
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

    '''
    os.environ['MASTER_ADDR'] = '165.246.44.232'  # it tells which IP address it should look for process 0
    os.environ['MASTER_PORT'] = '22'
    dist.init_process_group(backend='nccl', rank=1, world_size=2, init_method='env://')
    '''
    if torch.cuda.is_available():
        generator = torch.nn.DataParallel(generator, device_ids=[0, 1]).cuda()
        discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1]).cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
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
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' %
              (dt.now(), init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    # log_dir = output_dir + '/logs'
    log_dir = output_dir % 'logs'
    # ckpt_dir = output_dir + '/checkpoints'
    ckpt_dir = './Output/logs/checkpoints'
    train_writer = SummaryWriter('./output/logs/train')
    val_writer = SummaryWriter('./output/logs/test')

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils_GAN.AverageMeter()
        data_time = utils.network_utils_GAN.AverageMeter()
        generator_losses = utils.network_utils_GAN.AverageMeter()
        # generator_fake_losses = utils.network_utils_GAN.AverageMeter()
        discriminator_losses = utils.network_utils_GAN.AverageMeter()

        # switch models to training mode
        generator.train()
        discriminator.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes, ground_truth_volumes_mesh) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get random volume excepth matches to batch index
            '''
            r = list(itertools.chain(range(0, batch_idx), range(batch_idx + 1, n_batches)))
            fake_idx = random.choice(r)
            fake_volumes = train_data_loader.dataset[fake_idx][3]
            fake_volumes = torch.tensor(fake_volumes)
            '''

            # Get data from data loader
            ground_truth_volumes = utils.network_utils_GAN.var_or_cuda(ground_truth_volumes)
            # fake_volumes = utils.network_utils_GAN.var_or_cuda(fake_volumes)
            # ground_truth_volumes_mesh = utils.network_utils_GAN.var_or_cuda(ground_truth_volumes_mesh)
            rendering_images = utils.network_utils_GAN.var_or_cuda(rendering_images)

            ground_truth_volumes = ground_truth_volumes.float() / 255.
            # fake_volumes = fake_volumes.float() / 255.

            # Train Generator
            ground_truth_volumes = torch.squeeze(ground_truth_volumes)

            # code = torch.rand(512, 4, 4, 4)
            '''
            fake_code = Variable(torch.Tensor(np.random.normal(0, 1, (512, 4, 4, 4))))
            fake_code = utils.network_utils_GAN.var_or_cuda(fake_code)
            gen_fake_volumes = decoder(fake_code)
            gen_fake_volumes = gen_fake_volumes.float()
            gen_fake_volumes = torch.squeeze(gen_fake_volumes)

            generator_loss_fake = bce_loss(gen_fake_volumes, ground_truth_volumes)
            '''

            valid = Variable(torch.Tensor(ground_truth_volumes.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(torch.Tensor(ground_truth_volumes.size(0), 1).fill_(0.0), requires_grad=False)

            valid = utils.network_utils_GAN.var_or_cuda(valid)
            fake = utils.network_utils_GAN.var_or_cuda(fake)

            # Train Generator
            generator_solver.zero_grad()

            np.random.RandomState()
            code = Variable(torch.Tensor(np.random.normal(0, 1, size=(ground_truth_volumes.shape[0], 100))))
            gen_volumes = generator(code)
            dis_code = discriminator(gen_volumes)
            generator_loss = bce_loss(dis_code, valid)

            generator_loss.backward()
            generator_solver.step()

            # Train Discriminator
            discriminator.zero_grad()

            d_loss_real = bce_loss(discriminator(ground_truth_volumes), valid)
            d_loss_fake = bce_loss(discriminator(gen_volumes.detach()), fake)

            # discriminator loss
            discriminator_loss = (d_loss_real + d_loss_fake) / 2.

            discriminator_loss.backward()
            discriminator_solver.step()

            # Append loss to average metrics
            generator_losses.update(generator_loss.item())
            # generator_fake_losses.update(generator_loss_fake.item())
            discriminator_losses.update(discriminator_loss.item())

            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('Generator/BatchLoss', generator_loss.item(), n_itr)
            # train_writer.add_scalar('Generator_fake/BatchLoss', generator_loss_fake.item(), n_itr)
            train_writer.add_scalar('Discriminator/BatchLoss', discriminator_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            print(
                '[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) GLoss = %.8f DLoss = %.8f'
                % (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val,
                   data_time.val, generator_loss.item(), discriminator_loss.item()))



        # Append epoch loss to TensorBoard
        train_writer.add_scalar('Generator/EpochLoss', generator_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Discriminator/EpochLoss', discriminator_losses.avg, epoch_idx + 1)

        # Adjust learning rate
        generator_lr_scheduler.step()
        discriminator_lr_scheduler.step()

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) GLoss = %.8f DLoss = %.8f' %
              (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time,
               generator_losses.avg, discriminator_losses.avg))

        gv = gen_volumes.detach().cpu().numpy()
        rendering_views = utils.binvox_visualization.get_volume_views(gv,
                                                                      './output/image/test/gv',
                                                                      epoch_idx)
        np.save('./output/voxel/gv/gv_' + str(epoch_idx).zfill(6) + '.npy', gv)

        gtv = ground_truth_volumes.cpu().numpy()
        rendering_views = utils.binvox_visualization.get_volume_views(gtv,
                                                                      './output/image/test/gtv',
                                                                      epoch_idx)
        np.save('./output/voxel/gtv/gtv_' + str(epoch_idx).zfill(6) + '.npy', gtv)

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' %
                  (dt.now(), epoch_idx + 2, cfg.TRAIN.NUM_EPOCHES, n_views_rendering))

        # Validate the training models
        # iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, discriminator)
        # encoder_loss = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, refiner, merger)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils_GAN.save_checkpoints(cfg, './output/logs/checkpoints/ckpt-epoch-%04d.pth' % (epoch_idx + 1),
                                                 epoch_idx + 1, generator, generator_solver,
                                                 discriminator, discriminator_solver)

        # if iou > best_iou:
        '''
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_iou = iou
            # best_loss = encoder_loss
            best_epoch = epoch_idx + 1
            utils.network_utils_GAN.save_checkpoints(cfg, './output/logs/checkpoints/best-ckpt.pth',
                                                     epoch_idx + 1, generator, generator_solver,
                                                     discriminator, discriminator_solver,
                                                     best_iou, best_epoch)
                    
            print('[INFO] %s Best epoch [%d] / Best IoU [%.4f]' % (dt.now(), best_epoch, best_iou))
            # print('[INFO] %s Best epoch [%d] / Best Loss [%.4f]' % (dt.now(), best_epoch, best_loss))
            '''
    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
