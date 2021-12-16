import numpy as np
import os

res = 64
gtv_avg = np.zeros([res, res, res], dtype=float)

num_gtv = 0

for i in range (0, 58):
    if os.path.exists('J:/Program/Pix2Vox-master/voxel_gtv_log/npy/x_64_norm/gtv_f_' + str(i).zfill(7) + '.npy'):
        print(str(i) + ' exists (F)')
        x_save_load = np.load('J:/Program/Pix2Vox-master/voxel_gtv_log/npy/x_64_norm/gtv_f_' + str(i).zfill(7) + '.npy')

        gtv_avg[:, :, :] = gtv_avg[:, :, :] + x_save_load[:, :, :]
        num_gtv = num_gtv + 1

    if os.path.exists('J:/Program/Pix2Vox-master/voxel_gtv_log/npy/x_64_norm/gtv_m_' + str(i).zfill(7) + '.npy'):
        print(str(i) + ' exists (M)')
        x_save_load = np.load('J:/Program/Pix2Vox-master/voxel_gtv_log/npy/x_64_norm/gtv_m_' + str(i).zfill(7) + '.npy')

        gtv_avg[:, :, :] = gtv_avg[:, :, :] + x_save_load[:, :, :]
        num_gtv = num_gtv + 1

print(str(num_gtv) + ' Volume Founded')
if num_gtv > 0:
    gtv_avg[:, :, :] = gtv_avg[:, :, :] / float(num_gtv)
    np.save('J:/Program/Pix2Vox-master/voxel_gtv_log/npy/gtv_avg_x_64_norm.npy', gtv_avg)