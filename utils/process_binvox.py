import glob
import binvox_rw
import numpy as np
import math
import cv2

def cal_bb(type, bb_data):
    len_value = 0
    if type == 'x':
        for i in range(0, 32):
            if np.all(data[i, :, :] != 1.):
                len_value = i
                return len_value
    elif type == 'y':
        for i in range(0, 32):
            if np.all(data[:, i, :] != 1.):
                len_value = i
                return len_value
    elif type == 'z':
        for i in range(0, 32):
            if np.all(data[:, :, i] != 1.):
                len_value = i
                return len_value
    else:
        print('invalid type')
        return -1

    return 32


for i in range(1, 58):
    image_paths = glob.glob('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2/gtv_f_' + str(i).zfill(7) + '_a.binvox')
    if len(image_paths) > 0:
        with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2/gtv_f_' + str(i).zfill(7) + '_a.binvox', 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        data = model.data

        x_len = cal_bb('x', data)
        y_len = cal_bb('y', data)
        z_len = cal_bb('z', data)

        res_volume = np.zeros((32, 32, 32), dtype=np.uint8)

        # scale to fit (32x32x32)
        '''
        if x_len < 32:
            for j in range(0, 32):
                res_volume[j, :, :] = data[round(float(j) * float(x_len) / 32.), :, :] * 255

        if y_len < 32:
            for j in range(0, 32):
                res_volume[:, j, :] = data[:, round(float(j) * float(y_len) / 32.), :] * 255

        if z_len < 32:
            for j in range(0, 32):
                res_volume[:, :, j] = data[:, :, round(float(j) * float(z_len) / 32.)] * 255
                '''
        # Fill Inside
        '''
        for j in range(0, 31):
            arr = data[:, j, :].astype('uint8')
            arr_res = cv2.resize(arr, dsize=(320, 320), interpolation=cv2.INTER_AREA)
            grayImage = cv2.cvtColor(arr_res, cv2.COLOR_GRAY2BGR)

            cnts, hie = cv2.findContours(arr_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.fillPoly(grayImage, cnts, (255, 255, 255))

            arr_result = cv2.resize(grayImage, dsize=(32, 32), interpolation=cv2.INTER_AREA)
            arr_result = cv2.cvtColor(arr_result, cv2.COLOR_BGR2GRAY)

            res_volume[:, j, :] = arr_result
            '''
        # Thresholding
        remain_pos = np.where(data[:, :, :] >= 89)
        remove_pos = np.where(data[:, :, :] < 89)
        res_volume[remain_pos] = 255
        res_volume[remove_pos] = 0

        res_volume = res_volume.swapaxes(1, 2)

        voxels = binvox_rw.from_array(res_volume, [32, 32, 32], [0.0, 0.0, 0.0], 1, fix_coords=True)
        with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2_thres_0_35/gtv_f_' + str(i).zfill(3) + '_a.binvox', 'wb') as f:
            voxels.write(f)

        print(str(i) + ' finished')