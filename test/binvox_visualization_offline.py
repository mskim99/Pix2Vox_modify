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
for i in range (69, 400, 66):
    print(str(i) + ' started')
    data = np.load('J:/Program/Pix2Vox-master/voxel_log/211222_2_loss_GAN_train_epoch_0_400_G_1_D_linear_increase_10_r_img_2_L1_2_SSIM_4_IoU_drp_g_e_0_375_lr_1e-4_norm_res_128/gv/gv_' + str(i).zfill(6) + '.npy')

    volume = data.squeeze().__ge__(0.36)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.set_aspect('auto')
    ax.voxels(volume, edgecolor="k", linewidth=0.125)

    plt.savefig('J:/Program/Pix2Vox-master/image_log/211226_3_loss_GAN_test_epoch_400_G_1_D_linear_increase_10_r_img_2_L1_2_SSIM_4_IoU_drp_g_e_0_375_lr_1e-4_norm_res_128/gv_' + str(i).zfill(6) + '_IoU_rate_4.png', bbox_inches='tight')
    plt.close()
