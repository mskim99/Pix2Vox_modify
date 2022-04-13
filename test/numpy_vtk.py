from vtk.util import numpy_support
import vtk
import binvox_rw
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
    x_save_load = np.load('J:/Program/Pix2Vox-master/voxel_log/220407_4_loss_pix2vox_master_test_epoch_250_E_D_iou_6_0_weight_0_2_lr_1e-5_ct_vol_lol2_res_128/gv/gv_' + str(i).zfill(6) + '.npy')

    imdata = vtk.vtkImageData()

    depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(order='C'), deep=True, array_type=vtk.VTK_FLOAT)

    imdata.SetDimensions([128, 128, 128])
    # fill the vtk image data object
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(depthArray)

    if not os.path.isdir('J:/Program/Pix2Vox-master/voxel_log/220407_4_loss_pix2vox_master_test_epoch_250_E_D_iou_6_0_weight_0_2_lr_1e-5_ct_vol_lol2_res_128_mha'):
        os.mkdir('J:/Program/Pix2Vox-master/voxel_log/220407_4_loss_pix2vox_master_test_epoch_250_E_D_iou_6_0_weight_0_2_lr_1e-5_ct_vol_lol2_res_128_mha')
    else:
        print("OS not problem")

    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName('J:/Program/Pix2Vox-master/voxel_log/220407_4_loss_pix2vox_master_test_epoch_250_E_D_iou_6_0_weight_0_2_lr_1e-5_ct_vol_lol2_res_128_mha/gv_' + str(i).zfill(6) + '.mha')
    writer.SetInputData(imdata)
    writer.Write()

'''
with open('J:/Program/Pix2Vox-master/voxel_log/voxel_process/gv_value_0_2_res_128.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

data = model.data

imdata = vtk.vtkImageData()

depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

imdata.SetDimensions([128, 128, 128])
# fill the vtk image data object
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(model)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('J:/Program/Pix2Vox-master/voxel_log/voxel_process/gv_value_0_2_res_128.mha')
writer.SetInputData(imdata)
writer.Write()
'''