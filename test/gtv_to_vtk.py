import numpy as np
import cv2
import sys
import glob
import math
from datetime import datetime as dt
import binvox_rw
import math

from vtk.util import numpy_support
import vtk

res = 64

for i in range (1, 58):
    image_paths = glob.glob('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume/KISTI_Rendering/00000024/f_' + str(i).zfill(7) + '/rendering/a*.png')
    if len(image_paths) > 0:

        print('f ' + str(i) + ' exist')

        # gtv_volume_slices = np.zeros((22, len(image_paths), 28), dtype=np.uint8)
        gtv_volume_slices = np.zeros((res, len(image_paths), res), dtype=np.uint8)
        for idx, image_path in enumerate(image_paths):
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            rendering_image = cv2.resize(rendering_image, dsize=(res, res), interpolation=cv2.INTER_CUBIC)
            if len(rendering_image.shape) < 3:
                print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                        (dt.now(), image_path))
                sys.exit(2)

            # Thresholding
            '''
            ret, thr_img = cv2.threshold(rendering_image, 85, 255, cv2.THRESH_TOZERO)
            max_val = thr_img.max()
            gtv_volume_slices[:, idx, :] = thr_img[:, :, 0]
            gtv_volume_slices[:, idx, :] *= math.ceil(255. / float(max_val))
            '''

            gtv_volume_slices[:, idx, :] = rendering_image[:, :, 0]

        gtv_volume_slices = np.array(gtv_volume_slices, order='F')
        # gtv_volume_slices = gtv_volume_slices[:, :, :, 0]
        # gtv_volume_slices = np.resize(gtv_volume_slices, [22, len(image_paths), 28])

        res_volume = np.zeros((res, res, res), dtype=np.uint8)

        # if gtv_volume_slices.shape[1] < 32:
        # for j in range (0, vol_len):
            # res_volume[5:27, j, 2:30] = gtv_volume_slices[:, math.ceil(float(j) * float(vol_len) / float(len(image_paths))), :]
        for j in range(0, res):
            res_volume[:, j, :] = gtv_volume_slices[:, math.ceil(float(j) * float(len(image_paths) - 1) / float(res)), :]
        # res_volume[5:27, 0:len(image_paths), 2:30] = gtv_volume_slices[:, :, :]

        res_volume = res_volume.swapaxes(0, 2)

        voxels = binvox_rw.from_array(res_volume, [res, res, res], [0.0, 0.0, 0.0], 1, fix_coords=True)
        with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_64/gtv_f_' + str(i).zfill(7) + '_a.binvox', 'wb') as f:
            voxels.write(f)
'''
for i in range(0, 58):
    image_paths = glob.glob(
        'J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume/KISTI_Rendering/00000024/m_' + str(i).zfill(
            7) + '/rendering/a*.png')
    if len(image_paths) > 0:

        print('m ' + str(i) + ' exist')

        # gtv_volume_slices = np.zeros((22, len(image_paths), 28), dtype=np.uint8)
        gtv_volume_slices = np.zeros((32, len(image_paths), 32), dtype=np.uint8)
        for idx, image_path in enumerate(image_paths):
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            rendering_image = cv2.resize(rendering_image, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            if len(rendering_image.shape) < 3:
                print('[FATAL] %s It seems that there is something wrong with the image file %s' %
                      (dt.now(), image_path))
                sys.exit(2)

            ret, thr_img = cv2.threshold(rendering_image, 85, 255, cv2.THRESH_TOZERO)
            max_val = thr_img.max()
            gtv_volume_slices[:, idx, :] = thr_img[:, :, 0]
            gtv_volume_slices[:, idx, :] *= math.ceil(255. / float(max_val))

        gtv_volume_slices = np.array(gtv_volume_slices, order='F')
        # gtv_volume_slices = gtv_volume_slices[:, :, :, 0]
        # gtv_volume_slices = np.resize(gtv_volume_slices, [22, len(image_paths), 28])

        res_volume = np.zeros((32, 32, 32), dtype=np.uint8)
        # if gtv_volume_slices.shape[1] < 32:
        # for j in range(0, vol_len):
            # res_volume[5:27, j, 2:30] = gtv_volume_slices[:, math.ceil(float(j) * float(vol_len) / float(len(image_paths))), :]
        for j in range(0, 32):
            res_volume[:, j, :] = gtv_volume_slices[:, math.ceil(float(j) * float(vol_len) / float(len(image_paths))), :]
        # res_volume[5:27, 0:len(image_paths), 2:30] = gtv_volume_slices[:, :, :]

        voxels = binvox_rw.from_array(res_volume, [32, 32, 32], [0.0, 0.0, 0.0], 1, fix_coords=True)
        with open('J:/Program/Pix2Vox-master/voxel_gtv_log/gtv_m_' + str(i).zfill(7) + '_a_thr_85_norm.binvox', 'wb') as f:
            voxels.write(f)
            '''

