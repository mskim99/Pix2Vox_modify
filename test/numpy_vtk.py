from vtk.util import numpy_support
import vtk
import numpy as np
import cv2
import os
# from fill_holes import fill_holes

'''
for j in range (50, 150, 50):
    for i in range(0, 18):
        x_save_load = np.load(
            'J:/Program/Pix2Vox-master/voxel_log/211007_ct_32_vol_e_test_epoch_' + str(j) + '/gv_' + str(i).zfill(6) + '.npy')
        imdata = vtk.vtkImageData()
        # this is where the conversion happens
        depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(), deep=3)

        # fill the vtk image data object
        imdata.SetDimensions([32, 32, 32])
        imdata.SetSpacing([1, 1, 1])
        imdata.SetOrigin([0, 0, 0])
        imdata.GetPointData().SetScalars(depthArray)

        if not os.path.isdir('J:/Program/Pix2Vox-master/voxel_log/211007_ct_32_vol_e_test_epoch_' + str(j) + '_mha'):
            os.mkdir('J:/Program/Pix2Vox-master/voxel_log/211007_ct_32_vol_e_test_epoch_' + str(j) + '_mha')

        writer = vtk.vtkMetaImageWriter()
        writer.SetFileName(
            'J:/Program/Pix2Vox-master/voxel_log/211007_ct_32_vol_e_test_epoch_' + str(j) + '_mha/gv_mha_' + str(i).zfill(6) + '.mha')
        writer.SetInputData(imdata)
        writer.Write()

    print(str(j) + ' index ended')
    '''

for i in range (0, 400):
    x_save_load = np.load('J:/Program/Pix2Vox-master/voxel_log/211219_1_loss_GAN_train_epoch_0_400_G_1_D_linear_increase_10_r_img_3_L1_4_SSIM_2_IoU_0_3-0_5_drp_g_e_0_375_lr_5e-5_norm_res_128/gv/gv_' + str(i).zfill(6) + '.npy')

    imdata = vtk.vtkImageData()

    depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(order='C'), deep=True, array_type=vtk.VTK_FLOAT)

    imdata.SetDimensions([128, 128, 128])
    # fill the vtk image data object
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(depthArray)

    if not os.path.isdir('J:/Program/Pix2Vox-master/voxel_log/211219_1_loss_GAN_train_epoch_0_400_G_1_D_linear_increase_10_r_img_3_L1_4_SSIM_2_IoU_0_3-0_5_drp_g_e_0_375_lr_5e-5_norm_res_128_mha'):
        os.mkdir('J:/Program/Pix2Vox-master/voxel_log/211219_1_loss_GAN_train_epoch_0_400_G_1_D_linear_increase_10_r_img_3_L1_4_SSIM_2_IoU_0_3-0_5_drp_g_e_0_375_lr_5e-5_norm_res_128_mha')

    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName('J:/Program/Pix2Vox-master/voxel_log/211219_1_loss_GAN_train_epoch_0_400_G_1_D_linear_increase_10_r_img_3_L1_4_SSIM_2_IoU_0_3-0_5_drp_g_e_0_375_lr_5e-5_norm_res_128_mha/gv_' + str(i).zfill(6) + '.mha')
    writer.SetInputData(imdata)
    writer.Write()
