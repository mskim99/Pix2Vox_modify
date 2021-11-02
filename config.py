# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()
__C.DATASETS.SHAPENET                       = edict()

# ShapeNet Original
'''
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets/ShapeNet_simple.json'
# __C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH  = './datasets/PascalShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH        = './datasets/Shapenet/ShapeNetRendering/%s/%s/rendering/%02d.png'
# __C.DATASETS.SHAPENET.RENDERING_PATH      = '/home/hzxie/Datasets/ShapeNet/PascalShapeNetRendering/%s/%s/render_%04d.jpg'
__C.DATASETS.SHAPENET.VOXEL_PATH            = './datasets/Shapenet/ShapeNetVox32/%s/%s/model.binvox'
'''

# ShapeNet Modified (jzw)
'''
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = '/home/jzw/work/pix2vox/datasets/KISTI.json'
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/home/jzw/work/pix2vox/datasets/KISTI_volume_CT_mesh_sc64_fill/KISTI_Rendering/%s/%s/rendering/%s%03d.png'
__C.DATASETS.SHAPENET.RENDERING_VIEWS        = '/home/jzw/work/pix2vox/datasets/KISTI_volume_CT_mesh_sc64_fill/KISTI_Rendering/%s/%s/rendering/views.txt'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/home/jzw/work/pix2vox/datasets/KISTI_volume_CT_mesh_sc64_fill/KISTI_Vox/%s/%s/model.binvox'
__C.DATASETS.SHAPENET.VOXEL_MESH_PATH            = '/home/jzw/work/pix2vox/datasets/KISTI_volume_CT_mesh_sc64_fill/KISTI_Vox_BD/%s/%s/model.binvox'
'''

# ShapeNet Modified (AI Server)
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets/KISTI.json'
__C.DATASETS.SHAPENET.RENDERING_PATH        = './datasets/KISTI_volume_CT_mesh_sc32_fill/KISTI_Rendering/%s/%s/rendering/%s%03d.png'
__C.DATASETS.SHAPENET.RENDERING_VIEWS        = './datasets/KISTI_volume_CT_mesh_sc32_fill/KISTI_Rendering/%s/%s/rendering/views.txt'
__C.DATASETS.SHAPENET.VOXEL_PATH            = './datasets/KISTI_volume_CT_mesh_sc32_fill/KISTI_Vox32/%s/%s/model.binvox'
__C.DATASETS.SHAPENET.VOXEL_MESH_PATH            = './datasets/KISTI_volume_CT_mesh_sc32_fill/KISTI_Vox32_BD/%s/%s/model.binvox'


__C.DATASETS.PASCAL3D                       = edict()
__C.DATASETS.PASCAL3D.TAXONOMY_FILE_PATH    = './datasets/Pascal3D.json'
__C.DATASETS.PASCAL3D.ANNOTATION_PATH       = '/home/hzxie/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
__C.DATASETS.PASCAL3D.RENDERING_PATH        = '/home/hzxie/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG'
__C.DATASETS.PASCAL3D.VOXEL_PATH            = '/home/hzxie/Datasets/PASCAL3D/CAD/%s/%02d.binvox'
__C.DATASETS.PIX3D                          = edict()
__C.DATASETS.PIX3D.TAXONOMY_FILE_PATH       = './datasets/Pix3D.json'
__C.DATASETS.PIX3D.ANNOTATION_PATH          = '/home/hzxie/Datasets/Pix3D/pix3d.json'
__C.DATASETS.PIX3D.RENDERING_PATH           = '/home/hzxie/Datasets/Pix3D/img/%s/%s.%s'
__C.DATASETS.PIX3D.VOXEL_PATH               = '/home/hzxie/Datasets/Pix3D/model/%s/%s/%s.binvox'

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'ShapeNet'
__C.DATASET.TEST_DATASET                    = 'ShapeNet'
# __C.DATASET.TEST_DATASET                  = 'Pascal3D'
# __C.DATASET.TEST_DATASET                  = 'Pix3D'

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 112       # Image width for input
__C.CONST.IMG_H                             = 112       # Image height for input
__C.CONST.N_VOX                             = 32
__C.CONST.BATCH_SIZE                        = 1
__C.CONST.N_VIEWS_RENDERING                 = 1         # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_W                        = 128       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                        = 128       # Dummy property for Pascal 3D

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = 'D:/output'
__C.DIR.RANDOM_BG_PATH                      = '/home/hzxie/Datasets/SUN2012/JPEGImages'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.USE_REFINER                     = False
__C.NETWORK.USE_MERGER                      = True

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_WORKER                        = 0             # number of data workers (default:8)
__C.TRAIN.NUM_EPOCHES                       = 250
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .8
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER           = 0
__C.TRAIN.EPOCH_START_USE_MERGER            = 0
__C.TRAIN.ENCODER_LEARNING_RATE             = 1e-5
__C.TRAIN.DECODER_LEARNING_RATE             = 1e-6
__C.TRAIN.REFINER_LEARNING_RATE             = 1e-4
__C.TRAIN.MERGER_LEARNING_RATE              = 1e-4
__C.TRAIN.ENCODER_LR_MILESTONES             = [150]
__C.TRAIN.DECODER_LR_MILESTONES             = [150]
__C.TRAIN.REFINER_LR_MILESTONES             = [150]
__C.TRAIN.MERGER_LR_MILESTONES              = [150]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 25            # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = False

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
