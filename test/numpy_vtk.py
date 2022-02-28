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
    x_save_load = np.load('J:/Program/Pix2Vox-master/voxel_log/220221_3_loss_pix2vox_master_test_epoch_400_E_D_lr_1e-4_ct_vol_res_128/gv/gv_' + str(i).zfill(6) + '.npy')

    imdata = vtk.vtkImageData()

    depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(order='C'), deep=True, array_type=vtk.VTK_FLOAT)

    imdata.SetDimensions([128, 128, 128])
    # fill the vtk image data object
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(depthArray)


    if not os.path.isdir('J:/Program/Pix2Vox-master/voxel_log/220221_3_loss_pix2vox_master_test_epoch_400_E_D_lr_1e-4_ct_vol_res_128_mha'):
        os.mkdir('J:/Program/Pix2Vox-master/voxel_log/220221_3_loss_pix2vox_master_test_epoch_400_E_D_lr_1e-4_ct_vol_res_128_mha')

    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName('J:/Program/Pix2Vox-master/voxel_log/220221_3_loss_pix2vox_master_test_epoch_400_E_D_lr_1e-4_ct_vol_res_128_mha/gv_' + str(i).zfill(6) + '.mha')
    writer.SetInputData(imdata)
    writer.Write()
