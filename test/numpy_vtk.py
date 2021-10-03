from vtk.util import numpy_support
import vtk
import numpy as np
import cv2
import os
# from fill_holes import fill_holes
'''
for j in range (50, 300, 50):
    for i in range(0, 18):
        x_save_load = np.load(
            'I:/Program/Pix2Vox-master/voxel_log/210926_gv_32_vol_test_epoch_' + str(j) + '/gv_' + str(i).zfill(6) + '.npy')
        imdata = vtk.vtkImageData()
        # this is where the conversion happens
        depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(), deep=3)

        # fill the vtk image data object
        imdata.SetDimensions([32, 32, 32])
        imdata.SetSpacing([1, 1, 1])
        imdata.SetOrigin([0, 0, 0])
        imdata.GetPointData().SetScalars(depthArray)

        if not os.path.isdir('I:/Program/Pix2Vox-master/voxel_log/210926_gv_32_vol_test_epoch_2_' + str(j) + '_mha'):
            os.mkdir('I:/Program/Pix2Vox-master/voxel_log/210926_gv_32_vol_test_epoch_2_' + str(j) + '_mha')

        writer = vtk.vtkMetaImageWriter()
        writer.SetFileName(
            'I:/Program/Pix2Vox-master/voxel_log/210926_gv_32_vol_test_epoch_2_' + str(j) + '_mha/gv_mha_' + str(i).zfill(6) + '.mha')
        writer.SetInputData(imdata)
        writer.Write()

    print(str(j) + ' index ended')
'''
for i in range (0, 1):
    x_save_load = np.load('J:/Program/Pix2Vox-master/voxel_gtv_log/gtv_0000001_a.npy')
    print(x_save_load.shape)

    imdata = vtk.vtkImageData()

    depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

    imdata.SetDimensions([32, 32, 32])
    # fill the vtk image data object
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(depthArray)

    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName('J:/Program/Pix2Vox-master/voxel_gtv_log/mha/gtv_0000001_a.mha')
    writer.SetInputData(imdata)
    writer.Write()
