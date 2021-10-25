import binvox_rw
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

with open('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc32_fill/KISTI_Vox32_BD/00000024/f_0000001/model.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)
data = model.data

volume = data.squeeze().__ge__(102)
fig = plt.figure()
ax = fig.gca(projection=Axes3D.name)
ax.set_aspect('auto')
ax.voxels(volume, edgecolor="k")

plt.savefig('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc32_fill/KISTI_Vox32_BD/00000024/f_0000001/model.png', bbox_inches='tight')
plt.close()
