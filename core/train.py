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

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import utils.loss_function

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

import numpy as np

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
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)
    print('[DEBUG] %s Parameters in Encoder: %d.' % (dt.now(), utils.network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(refiner)))
    print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), utils.network_utils.count_parameters(merger)))

    # Initialize weights of networks
    encoder.apply(utils.network_utils.init_weights)
    decoder.apply(utils.network_utils.init_weights)
    refiner.apply(utils.network_utils.init_weights)
    merger.apply(utils.network_utils.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                          lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS
                                          , weight_decay=0.2
                                          )
        decoder_solver = torch.optim.Adam(decoder.parameters(),
                                          lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS
                                          , weight_decay=0.2
                                          )
        refiner_solver = torch.optim.Adam(refiner.parameters(),
                                          lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                          betas=cfg.TRAIN.BETAS
                                          , weight_decay=0.2
                                          )
        merger_solver = torch.optim.Adam(merger.parameters(),
                                         lr=cfg.TRAIN.MERGER_LEARNING_RATE,
                                         betas=cfg.TRAIN.BETAS
                                         , weight_decay=0.2
                                         )
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM
                                         , weight_decay=0.2
                                         )
        decoder_solver = torch.optim.SGD(decoder.parameters(),
                                         lr=cfg.TRAIN.DECODER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM
                                         , weight_decay=0.2
                                         )
        refiner_solver = torch.optim.SGD(refiner.parameters(),
                                         lr=cfg.TRAIN.REFINER_LEARNING_RATE,
                                         momentum=cfg.TRAIN.MOMENTUM
                                         , weight_decay=0.2
                                         )
        merger_solver = torch.optim.SGD(merger.parameters(),
                                        lr=cfg.TRAIN.MERGER_LEARNING_RATE,
                                        momentum=cfg.TRAIN.MOMENTUM
                                        , weight_decay=0.2
                                        )
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                milestones=cfg.TRAIN.ENCODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                milestones=cfg.TRAIN.DECODER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver,
                                                                milestones=cfg.TRAIN.REFINER_LR_MILESTONES,
                                                                gamma=cfg.TRAIN.GAMMA)
    merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(merger_solver,
                                                               milestones=cfg.TRAIN.MERGER_LR_MILESTONES,
                                                               gamma=cfg.TRAIN.GAMMA)
    '''
    os.environ['MASTER_ADDR'] = '165.246.44.232'  # it tells which IP address it should look for process 0
    os.environ['MASTER_PORT'] = '22'
    dist.init_process_group(backend='nccl', rank=1, world_size=2, init_method='env://')
    '''
    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder, device_ids=[0, 1], output_device=1).cuda()
        decoder = torch.nn.DataParallel(decoder, device_ids=[0, 1], output_device=0).cuda()
        refiner = torch.nn.DataParallel(refiner, device_ids=[0, 1], output_device=0).cuda()
        merger = torch.nn.DataParallel(merger, device_ids=[0, 1], output_device=0).cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()
    # ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    # l1_loss = torch.nn.L1Loss()
    # smooth_l1_loss = torch.nn.SmoothL1Loss()
    # huber_loss = torch.nn.HuberLoss(delta=0.5)

    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_loss = 10000000000
    best_epoch = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

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
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        gtv_losses = utils.network_utils.AverageMeter()
        gtvm_losses = utils.network_utils.AverageMeter()
        encoder_losses = utils.network_utils.AverageMeter()
        refiner_losses = utils.network_utils.AverageMeter()

        # switch models to training mode
        # print('[DEBUG]  Training...')
        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()
        # print('[DEBUG]  END Training')

        # print('[DEBUG]  Start Measurement...')
        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images,
                        ground_truth_volumes, ground_truth_volumes_mesh) in enumerate(train_data_loader):
            # Measure data time
            data_time.update(time() - batch_end_time)

            # Create Ground truth volume using Axis of CT Volume
            '''
            ground_truth_volumes = rendering_images[0]

            ground_truth_volumes = ground_truth_volumes[0, :, :, :, 0]
            ground_truth_volumes = np.resize(ground_truth_volumes, [1, 32, 32, 32])
            ground_truth_volumes = torch.Tensor(ground_truth_volumes)
            ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)

            # For debugging
            gtv_arr = ground_truth_volumes.cpu().numpy()
            gtv_arr = gtv_arr[0, :, :, :]

            np.save('/home/jzw/work/pix2vox/output/voxel/gtv_log/Pix2Vox-masterf_' + str(batch_idx) + '_a', gtv_arr)
            '''
            # Get data from data loader
            ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)
            ground_truth_volumes_mesh = utils.network_utils.var_or_cuda(ground_truth_volumes_mesh)
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)

            # Train the encoder, decoder, refiner, and merger
            image_features = encoder(rendering_images)
            raw_features, generated_volumes = decoder(image_features)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volumes = merger(raw_features, generated_volumes)
            else:
                generated_volumes = torch.mean(generated_volumes, dim=1)

            generated_volumes = generated_volumes.float()
            ground_truth_volumes = ground_truth_volumes.float() / 255.
            ground_truth_volumes_mesh = ground_truth_volumes_mesh.float() / 255.

            # Set target for Cross Entropy Loss (4 classes)
            # 1 : 0 ~ 73 / 2 : 73 ~ 81, 3 : 81 ~ 97, 4 : 97 ~ 255
            '''
            class1_pos = torch.where(ground_truth_volumes[:, :, :] < 0.2863)
            class2_pos = torch.where((ground_truth_volumes[:, :, :] >= 0.2863) & (ground_truth_volumes[:, :] < 0.3176))
            class3_pos = torch.where((ground_truth_volumes[:, :, :] >= 0.3176) & (ground_truth_volumes[:, :] < 0.3804))
            class4_pos = torch.where(ground_truth_volumes[:, :, :] >= 0.3804)

            gtv_target = torch.zeros(ground_truth_volumes.shape).cuda()
            gtv_target[class1_pos] = 1.0
            gtv_target[class2_pos] = 2.0
            gtv_target[class3_pos] = 3.0
            gtv_target[class4_pos] = 4.0

            bce_logits_loss = torch.nn.BCEWithLogitsLoss(pos_weight=gtv_target)
            '''
            '''
            loss_iou_thres = [0.2, 0.3, 0.4, 0.5]
            loss_iou_weights = [2.0, 1.0, 1.0, 2.0]
            loss_ious = 0.0
            for index in range(0, 4):
                _volume = torch.ge(generated_volumes, loss_iou_thres[index]).float()
                _gt_volume = torch.ge(ground_truth_volumes, loss_iou_thres[index]).float()
                intersection = torch.sum(torch.ge(_volume.mul(_gt_volume), 1)).float()
                union = torch.sum(torch.ge(_volume.add(_gt_volume), 1)).float()
                part_loss_iou = 0.0
                if index == 0 | index == 1:
                    part_loss_iou = loss_iou_weights[index] * (intersection / union)
                elif index == 2 | index == 3:
                    part_loss_iou = loss_iou_weights[index] * (1. - (intersection / union))
                loss_ious = loss_ious + part_loss_iou
                '''
            # print("gv size : " + str(generated_volumes.size()))
            # print("gtv size : " + str(ground_truth_volumes.size()))
            # encoder_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10
            gtv_loss = mse_loss(generated_volumes, ground_truth_volumes) * 300
            # encoder_loss = utils.loss_function.loss_gtv(generated_volumes, ground_truth_volumes, 0.4, 0.5, 0.5) * 300
            # encoder_loss = l1_loss(generated_volumes, ground_truth_volumes) * 10
            # encoder_loss = smooth_l1_loss(generated_volumes, ground_truth_volumes) * 300
            # encoder_loss = huber_loss(generated_volumes, ground_truth_volumes) * 900
            # old_loss = ce_loss(generated_volumes, gtv_target) * 10
            # old_loss = bce_logits_loss(generated_volumes, gtv_target) * 30
            # encoder_loss = 3e12 * utils.loss_function.ls_loss(generated_volumes, ground_truth_volumes, 0.3137, 1.)
            # encoder_loss = utils.loss_function.dice_loss(generated_volumes, ground_truth_volumes, 0.4, 1.0, 30.)
            gtvm_loss = bce_loss(generated_volumes, ground_truth_volumes_mesh) * 10
            encoder_loss = (gtv_loss + gtvm_loss) / 2.

            '''
            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volumes = refiner(generated_volumes)
                # refiner_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10
                # refiner_loss = mse_loss(generated_volumes, ground_truth_volumes) * 300
                # refiner_loss = utils.loss_function.loss_gtv(generated_volumes, ground_truth_volumes, 0.4, 0.5, 0.5) * 300
                # refiner_loss = l1_loss(generated_volumes, ground_truth_volumes) * 10
                # refiner_loss = smooth_l1_loss(generated_volumes, ground_truth_volumes) * 300
                # refiner_loss = huber_loss(generated_volumes, ground_truth_volumes) * 900

            else:
            '''
            refiner_loss = encoder_loss

            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()
            refiner.zero_grad()
            merger.zero_grad()

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                encoder_loss.backward(retain_graph=True)
                refiner_loss.backward()
            else:
                encoder_loss.backward()

            encoder_solver.step()
            decoder_solver.step()
            refiner_solver.step()
            merger_solver.step()

            # Append loss to average metrics
            gtv_losses.update(gtv_loss.item())
            gtvm_losses.update(gtvm_loss.item())
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
            train_writer.add_scalar('Refiner/BatchLoss', refiner_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            print(
                '[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) VLoss = %.4f MLoss = %.4f EDLoss = %.4f'
                % (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time.val,
                   data_time.val, gtv_loss.item(), gtvm_loss.item(), encoder_loss.item()))

        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        refiner_lr_scheduler.step()
        merger_lr_scheduler.step()

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) VLoss = %.4f MLoss = %.4f EDLoss = %.4f' %
              (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, gtv_losses.avg, gtvm_losses.avg, encoder_losses.avg))

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' %
                  (dt.now(), epoch_idx + 2, cfg.TRAIN.NUM_EPOCHES, n_views_rendering))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, refiner, merger)
        # encoder_loss = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, refiner, merger)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg, '/home/jzw/work/pix2vox/output/logs/checkpoints/ckpt-epoch-%04d.pth' % (epoch_idx + 1),
                                                 epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver,
                                                 refiner, refiner_solver, merger, merger_solver, best_iou, best_epoch)

        # if iou > best_iou:
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_iou = iou
            # best_loss = encoder_loss
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg, '/home/jzw/work/pix2vox/output/logs/checkpoints/best-ckpt.pth', epoch_idx + 1, encoder,
                                                 encoder_solver, decoder, decoder_solver, refiner, refiner_solver,
                                                 merger, merger_solver, best_iou, best_epoch)

            print('[INFO] %s Best epoch [%d] / Best IoU [%.4f]' % (dt.now(), best_epoch, best_iou))
            # print('[INFO] %s Best epoch [%d] / Best Loss [%.4f]' % (dt.now(), best_epoch, best_loss))

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()
