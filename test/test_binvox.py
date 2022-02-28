import binvox_rw
import numpy as np
import sys
import math
import cv2
np.set_printoptions(threshold=sys.maxsize)

with open('J:/program/Pix2Vox-origin/datasets/KISTI_volume_CT_mesh_sc128_outer_ct/KISTI_Vox/00000024/f_0000001/model.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

data = model.data
'''
x_len = -1
y_len = -1
z_len = -1

def cal_bb(type):
    len_value = 0
    if type == 'x':
        for i in range(0, 32):
            if np.all(data[i, :, :] != 255):
                len_value = i
                return len_value
    elif type == 'y':
        for i in range(0, 32):
            if np.all(data[:, i, :] != 255):
                len_value = i
                return len_value
    elif type == 'z':
        for i in range(0, 32):
            if np.all(data[:, :, i] != 255):
                len_value = i
                return len_value
    else:
        print('invalid type')
        return -1

    return 32
'''

# data = np.load('J:/Program/Pix2Vox-master/voxel_log/voxel_process/gv_000249.npy')

for i in range(0, 127):
    for j in range(0, 127):
        sys.stdout.write(str(data[63, i, j]))
        sys.stdout.write(' ')
    print('')
print('')

'''
filled_arr = np.zeros([32, 32, 32])

for i in range(0, 31):
    arr = data[:, i, :].astype('uint8')
    arr_res = cv2.resize(arr, dsize=(320, 320), interpolation=cv2.INTER_AREA)
    grayImage = cv2.cvtColor(arr_res, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('J:/DK_Data_Process/i_1-3_Target_Mesh/fill_inner/source/source' + str(i) + '.jpg', grayImage)

    grayImage_cont = grayImage.copy()
    cnts, hie = cv2.findContours(arr_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        cv2.drawContours(grayImage_cont, [cnt], 0, (255, 0, 0), 3)  # blue
    cv2.imwrite('J:/DK_Data_Process/i_1-3_Target_Mesh/fill_inner/contour/contour' + str(i) + '.jpg', grayImage_cont)

    cv2.fillPoly(grayImage, cnts, (255, 255, 255))
    cv2.imwrite('J:/DK_Data_Process/i_1-3_Target_Mesh/fill_inner/result/result' + str(i) + '.jpg', grayImage)

    arr_result = cv2.resize(grayImage, dsize=(32, 32), interpolation=cv2.INTER_AREA)
    arr_result = cv2.cvtColor(arr_result, cv2.COLOR_BGR2GRAY)

    filled_arr[:, i, :] = arr_result
    '''
'''
for i in range(0, 31):
    for j in range(0, 31):
        sys.stdout.write(str(math.ceil(filled_arr[i, j, 11] / 255)))
        sys.stdout.write(' ')
    print('')
'''

'''
arr = cv2.resize(arr, dsize=(256, 256), interpolation=cv2.INTER_AREA)
kernel = np.ones((21, 21), np.uint8)
result = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)

grayImage = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
dst = cv2.resize(grayImage, dsize=(320, 320), interpolation=cv2.INTER_AREA)
cv2.imshow("Result", dst)
'''
'''
for i in range(0, 31):
    kernel = np.ones((11, 11), np.uint8)
    result = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel)
    '''
