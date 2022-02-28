import glob
import binvox_rw
import numpy as np
import math
import cv2

res = 128

def cal_bb(type, bb_data):
    len_value = 0
    if type == 'x':
        for i in range(0, res):
            if np.all(data[i, :, :] != 1.):
                len_value = i
                return len_value
    elif type == 'y':
        for i in range(0, res):
            if np.all(data[:, i, :] != 1.):
                len_value = i
                return len_value
    elif type == 'z':
        for i in range(0, res):
            if np.all(data[:, :, i] != 1.):
                len_value = i
                return len_value
    else:
        print('invalid type')
        return -1

    return res


for i in range(1, 58):
    image_paths = glob.glob('J:/DK_Data_Process/i_1-3_Target_Mesh/23_24_spine/res_128/f_' + str(i).zfill(3) + '_vrt_23_24_res_128.binvox')
    if len(image_paths) > 0:
        with open('J:/DK_Data_Process/i_1-3_Target_Mesh/23_24_spine/res_128/f_' + str(i).zfill(3) + '_vrt_23_24_res_128.binvox', 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        data = model.data

        x_len = cal_bb('x', data)
        y_len = cal_bb('y', data)
        z_len = cal_bb('z', data)

        res_volume = np.zeros((res, res, res), dtype=np.uint8)

        # scale to fit
        if x_len < res:
            for j in range(0, res):
                res_volume[j, :, :] = data[round(float(j) * float(x_len) / float(res)), :, :] * 255

        if y_len < res:
            for j in range(0, res):
                res_volume[:, j, :] = data[:, round(float(j) * float(y_len) / float(res)), :] * 255

        if z_len < res:
            for j in range(0, res):
                res_volume[:, :, j] = data[:, :, round(float(j) * float(z_len) / float(res))] * 255

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
        '''
        remain_pos = np.where(data[:, :, :] >= 89)
        remove_pos = np.where(data[:, :, :] < 89)
        res_volume[remain_pos] = 255
        res_volume[remove_pos] = 0
        '''

        res_volume = res_volume.swapaxes(1, 2)

        voxels = binvox_rw.from_array(res_volume, [res, res, res], [0.0, 0.0, 0.0], 1, fix_coords=True)
        with open('J:/DK_Data_Process/i_1-3_Target_Mesh/23_24_spine/res_128_fit/f_' + str(i).zfill(3) + '_vrt_23_24_res_128.binvox', 'wb') as f:
            voxels.write(f)

        print(str(i) + ' finished')