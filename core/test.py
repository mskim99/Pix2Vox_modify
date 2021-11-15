# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

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
import utils.extract_amplify_features as uaf

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger

import cv2

from PIL import Image

def test_net(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None):
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
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger = Merger(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            refiner = torch.nn.DataParallel(refiner).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

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
    gtv_mse_losses = utils.network_utils.AverageMeter()
    gtv_dice_losses = utils.network_utils.AverageMeter()
    gtv_losses = utils.network_utils.AverageMeter()
    gtvm_losses = utils.network_utils.AverageMeter()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    # Calculate ground truth volumes thresholding
    ground_truth_volumes_thres_test = torch.zeros([len(test_data_loader), 1, 64, 64, 64])
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume, ground_truth_volume_mesh) in enumerate(test_data_loader):

        ground_truth_volume = ground_truth_volume.float() / 255.
        ground_truth_volume_thres = uaf.Extract_Amplify_Features(ground_truth_volume, 0.36, 64)
        ground_truth_volumes_thres_test[sample_idx] = ground_truth_volume_thres

        gtv = ground_truth_volume.cpu().numpy()
        gtvt = ground_truth_volume_thres.cpu().numpy()

        if output_dir and sample_idx == 0:
            np.save('./output/voxel/gtv/gtv_' + str(epoch_idx).zfill(6) + '.npy', gtv)
            np.save('./output/voxel/gtvt/gtvt_' + str(epoch_idx).zfill(6) + '.npy', gtvt)

    ground_truth_volumes_thres_test = utils.network_utils.var_or_cuda(ground_truth_volumes_thres_test)

    vol_write_idx = 0
    min_loss = 10000000.0
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume, ground_truth_volume_mesh) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            # Get data from data loader

            ground_truth_volume = ground_truth_volume.float() / 255.
            ground_truth_volume = uaf.Extract_Features_torch(ground_truth_volume, 0.29, 64)

            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            raw_features, generated_volume = decoder(image_features)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)

            generated_volume = generated_volume.float()
            # ground_truth_volume_thres = uaf.Extract_Amplify_Features(ground_truth_volume, 0.35, 32)
            # ground_truth_volume_mesh = ground_truth_volume_mesh.float() / 255.

            # encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10
            # encoder_loss = loss_ious * mse_loss(generated_volume, ground_truth_volume) * 100
            # encoder_loss = utils.loss_function.loss_gtv(generated_volume, ground_truth_volume, 0.4, 0.5, 0.5) * 300
            # encoder_loss = l1_loss(generated_volume, ground_truth_volume) * 10
            # encoder_loss = smooth_l1_loss(generated_volume, ground_truth_volume) * 300
            # encoder_loss = huber_loss(generated_volume, ground_truth_volume) * 900
            # old_loss = ce_loss(generated_volume, gtv_target) * 10
            # old_loss = bce_logits_loss(generated_volume, gtv_target) * 30
            # new_loss = 3e12 * utils.loss_function.ls_loss(generated_volume, ground_truth_volume, 0.3137, 1.)
            # + new_loss
            # encoder_loss = utils.loss_function.dice_loss(generated_volume, ground_truth_volume, 0.4, 1.0, 30.)
            gv_p = torch.ge(generated_volume, 0.29).float() * generated_volume
            gtv_p = torch.ge(ground_truth_volume, 0.29).float() * ground_truth_volume
            gtv_mse_loss = mse_loss(gv_p, gtv_p) * 50.
            gtv_dice_loss = utils.loss_function.dice_loss_weight(generated_volume, ground_truth_volume, 0.22, 0.36) * 30.
            gtv_loss = (gtv_mse_loss + gtv_dice_loss) / 2.
            gtvm_loss = bce_loss(generated_volume, ground_truth_volumes_thres_test[sample_idx]) * 30.
            encoder_loss = (gtvm_loss + gtv_loss) * 0.5

            '''
            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refiner(generated_volume)
                # refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10
                refiner_loss = mse_loss(generated_volume, ground_truth_volume) * 300
                # refiner_loss = utils.loss_function.loss_gtv(generated_volume, ground_truth_volume, 0.4, 0.5, 0.5) * 300
                # refiner_loss = l1_loss(generated_volume, ground_truth_volume) * 10
                # refiner_loss = smooth_l1_loss(generated_volume, ground_truth_volume) * 300
                # refiner_loss = huber_loss(generated_volume, ground_truth_volume) * 900

            else:
            '''
            refiner_loss = encoder_loss

            # Append loss and accuracy to average metrics
            gtv_mse_losses.update(gtv_mse_loss.item())
            gtv_dice_losses.update(gtv_dice_loss.item())
            gtv_losses.update(gtv_loss.item())
            gtvm_losses.update(gtvm_loss.item())
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            # Volume Visualization
            gv = generated_volume.cpu().numpy()
            '''
            rendering_views = utils.binvox_visualization.get_volume_views(gv, '/home/jzw/work/pix2vox/output/image/test/gv',
                                                        vol_write_idx)
                                                        '''

            # np.save('./output/voxel_test/gv/gv_' + str(sample_idx).zfill(6) + '.npy', gv)


            # IoU per sample
            sample_iou = []
            sample_dice = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
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
            if output_dir and sample_idx == 0:
                # img_dir = output_dir % 'images'
                # Volume Visualization
                gv = generated_volume.cpu().numpy()
                np.save('./output/voxel/gv/gv_' + str(epoch_idx).zfill(6) + '.npy', gv)

                '''
                if epoch_idx % 25 == 0:
                    
                    rendering_views = utils.binvox_visualization.get_volume_views(gv, '/home/jzw/work/pix2vox/output/image/test/gv',
                                                                                  epoch_idx)
                    
                    # print(np.shape(rendering_views))
                    # rendering_views_im = np.array((rendering_views * 255), dtype=np.uint8)
                    # test_writer.add_image('Test Sample#%02d/Volume Reconstructed' % sample_idx, rendering_views_im, epoch_idx)
                    gtvm = ground_truth_volume_thres.cpu().numpy()
                    rendering_views = utils.binvox_visualization.get_volume_views(gtvm, '/home/jzw/work/pix2vox/output/image/test/gtvm',
                                                                                  epoch_idx)
                    # rendering_views_im = np.array((rendering_views * 255), dtype=np.uint8)
                    # test_writer.add_image('Test Sample#%02d/Volume GroundTruth' % sample_idx, rendering_views_im, epoch_idx)
                    gtv = ground_truth_volume.cpu().numpy()
                    rendering_views = utils.binvox_visualization.get_volume_views(gtv,
                                                                                  '/home/jzw/work/pix2vox/output/image/test/gtv',
                                                                                  epoch_idx)
                                                                                  '''

            # Print sample loss('IoU = %s' removed)
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s V_L1Loss = %.4f V_DLoss = %.4f VLoss = %.4f MLoss = %.4f EDLoss = %.4f'
                  % (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, gtv_mse_loss.item(), gtv_dice_loss.item(), gtv_loss.item(),
                     gtvm_loss.item(), encoder_loss.item(),
                     ))

            if encoder_loss < min_loss:
                min_loss = encoder_loss

    print('[INFO] %s Test[%d] Loss Mean / V_L1Loss = %.4f V_DLoss = %.4f VLoss = %.4f MLoss = %.4f EDLoss = %.4f'
          % (dt.now(), n_samples, gtv_mse_losses.avg, gtv_dice_losses.avg, gtv_losses.avg, gtvm_losses.avg, encoder_losses.avg))

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
