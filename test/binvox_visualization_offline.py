import binvox_rw
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

'''
with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_64_thres_0_35/gtv_f_001_a.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)
data = model.data
'''

data = np.load('J:/Program/Pix2Vox-master/voxel_log/211102_1_ct_vol_e_pix2vox_test_m_BCE_15_lr_1e-5_v_dice_30_sc64_fill_lr_1e-4_4_epoch_300_thr_0_4_max_h/gv_000004.npy')

volume = data.squeeze().__ge__(0.4)
fig = plt.figure()
ax = fig.gca(projection=Axes3D.name)
ax.set_aspect('auto')
ax.voxels(volume, edgecolor="k", linewidth=0.25)

plt.savefig('J:/Program/Pix2Vox-master/image_log/211102_1_ct_vol_e_pix2vox_train_m_BCE_15_lr_1e-5_v_dice_30_sc64_fill_lr_1e-4_4_epoch_0_400_thr_0_4_max_h/gv_000004_epoch_300.png', bbox_inches='tight')
plt.close()
