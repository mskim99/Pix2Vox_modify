import numpy as np
import glob
import binvox_rw
import os

res = 64

# Normalize binvox
'''
for i in range(1, 56):
    if os.path.isfile('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_128_23_24/gtv_f_' + str(i).zfill(7) + '_a.binvox'):
        with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_128_23_24/gtv_f_' + str(i).zfill(7) + '_a.binvox', 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        data = model.data

        res_volume = data.astype('float')
        res_volume = res_volume / 255.

        np.save('J:/Program/Pix2Vox-master/voxel_gtv_log/npy/x_128_23_24_norm/gtv_f_' + str(i).zfill(7) + '.npy', res_volume)

        print(str(i) + ' finished (f)')

for i in range(1, 58):
    if os.path.isfile('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_128_23_24/gtv_m_' + str(i).zfill(7) + '_a.binvox'):
        with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_128_23_24/gtv_m_' + str(i).zfill(7) + '_a.binvox', 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        data = model.data
    
        res_volume = data.astype('float')
        res_volume = res_volume / 255.

        np.save('J:/Program/Pix2Vox-master/voxel_gtv_log/npy/x_128_23_24_norm/gtv_m_' + str(i).zfill(7) + '.npy', res_volume)

        print(str(i) + ' finished (m)')
        '''

npy_paths = glob.glob('J:/Program/Pix2Vox-master/voxel_gtv_log/npy/x_128_23_24_norm/*.npy')
num_npy = len(npy_paths)
prob = np.zeros(101)
print(prob.shape)
if num_npy > 0:
    for path in npy_paths:

        data = np.load(path)

        for i in range(0, 100):
            prob_pos = np.where((data[:, :, :] >= (float(i) / 100.)) & (data[:, :, :] < (float(i + 1) / 100.)))
            prob_num = len(prob_pos[0])
            prob[i] += prob_num

        prob_pos = np.where(data[:, :, :] == 1.)
        prob_num = len(prob_pos[0])
        prob[100] += prob_num

for i in range(0, 100):
    print('%d' % (prob[i]))
print('%d' % (prob[100]))
print('Sum : ' + str(np.sum(prob)))

