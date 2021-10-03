import numpy as np
import cv2
import sys
import glob
import math
from datetime import datetime as dt
import binvox_rw

from vtk.util import numpy_support
import vtk

for i in range (0, 58):
    image_paths = glob.glob('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/a*.png')
    if len(image_paths) > 0:

        print('f ' + str(i) + ' exist')

        orig_shape_x = -1
        orig_shape_y = -1

        gtv_volume_slices = []
        for image_path in image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            orig_shape_x = rendering_image.shape[0]
            orig_shape_y = rendering_image.shape[1]
            rendering_image = cv2.resize(rendering_image, dsize=(32, 32))
            if len(rendering_image.shape) < 3:
                print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                        (dt.now(), image_path))
                sys.exit(2)

            gtv_volume_slices.append(rendering_image)

        gtv_volume_slices = np.array(gtv_volume_slices, 'f')
        gtv_volume_slices = gtv_volume_slices[:, :, :, 0]
        gtv_volume_slices = np.resize(gtv_volume_slices, [32, 32, 32])

        voxels = binvox_rw.from_array(gtv_volume_slices, [32, 32, 32], [0.0, 0.0, 0.0], 1, fix_coords=True)
        with open('J:/Program/Pix2Vox-master/voxel_gtv_log/gtv_f_' + str(i).zfill(7) + '_a.binvox', 'wb') as f:
            voxels.write(f)

for i in range(0, 58):
    image_paths = glob.glob('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume/KISTI_Rendering/00000024/m_' + str(i).zfill(7) + '/rendering/a*.png')
    if len(image_paths) > 0:

        print('m ' + str(i) + ' exist')

        orig_shape_x = -1
        orig_shape_y = -1

        gtv_volume_slices = []
        for image_path in image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            orig_shape_x = rendering_image.shape[0]
            orig_shape_y = rendering_image.shape[1]
            rendering_image = cv2.resize(rendering_image, dsize=(32, 32))
            if len(rendering_image.shape) < 3:
                print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                      (dt.now(), image_path))
                sys.exit(2)

            gtv_volume_slices.append(rendering_image)

        gtv_volume_slices = np.array(gtv_volume_slices, 'f')
        gtv_volume_slices = gtv_volume_slices[:, :, :, 0]
        gtv_volume_slices = np.resize(gtv_volume_slices, [32, 32, 32])

        voxels = binvox_rw.from_array(gtv_volume_slices, [32, 32, 32], [0.0, 0.0, 0.0], 1, fix_coords=True)
        with open('J:/Program/Pix2Vox-master/voxel_gtv_log/gtv_m_' + str(i).zfill(7) + '_a.binvox', 'wb') as f:
            voxels.write(f)
