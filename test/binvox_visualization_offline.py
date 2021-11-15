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

data = np.load('J:/Program/Pix2Vox-master/voxel_log/211110_5_loss_GAN_epoch_0_250_G_1_D_linear_increase_7/gv/gv_000249.npy')

volume = data.squeeze().__ge__(0.36)
fig = plt.figure()
ax = fig.gca(projection=Axes3D.name)
ax.set_aspect('auto')
ax.voxels(volume, edgecolor="k", linewidth=0.25)

plt.savefig('J:/Program/Pix2Vox-master/image_log/211110_5_loss_GAN_epoch_0_250_G_1_D_linear_increase_7/gv_000249.png', bbox_inches='tight')
plt.close()
