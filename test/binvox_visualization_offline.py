import binvox_rw
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_64_thres_0_35/gtv_f_001_a.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)
data = model.data

volume = data.squeeze().__ge__(102)
fig = plt.figure()
ax = fig.gca(projection=Axes3D.name)
ax.set_aspect('auto')
ax.voxels(volume, edgecolor="k")

plt.savefig('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_64_thres_0_35/gtv_f_001_a.png', bbox_inches='tight')
plt.close()
