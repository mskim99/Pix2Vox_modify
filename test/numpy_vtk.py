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

for i in range (0, 18):
    x_save_load = np.load('J:/Program/Pix2Vox-master/voxel_log/211102_1_ct_vol_e_pix2vox_test_m_BCE_15_lr_1e-5_v_dice_30_sc64_fill_lr_1e-4_4_epoch_375_thr_0_4_max_h/gv_' + str(i).zfill(6) + '.npy')

    imdata = vtk.vtkImageData()

    depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(order='C'), deep=True, array_type=vtk.VTK_FLOAT)

    imdata.SetDimensions([64, 64, 64])
    # fill the vtk image data object
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(depthArray)

    if not os.path.isdir('J:/Program/Pix2Vox-master/voxel_log/211102_1_ct_vol_e_pix2vox_test_m_BCE_15_lr_1e-5_v_dice_30_sc64_fill_lr_1e-4_4_epoch_375_thr_0_4_max_h_mha'):
        os.mkdir('J:/Program/Pix2Vox-master/voxel_log/211102_1_ct_vol_e_pix2vox_test_m_BCE_15_lr_1e-5_v_dice_30_sc64_fill_lr_1e-4_4_epoch_375_thr_0_4_max_h_mha')

    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName('J:/Program/Pix2Vox-master/voxel_log/211102_1_ct_vol_e_pix2vox_test_m_BCE_15_lr_1e-5_v_dice_30_sc64_fill_lr_1e-4_4_epoch_375_thr_0_4_max_h_mha/gv_' + str(i).zfill(6) + '.mha')
    writer.SetInputData(imdata)
    writer.Write()
