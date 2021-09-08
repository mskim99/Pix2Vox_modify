from vtk.util import numpy_support
import vtk
import numpy as np

'''
for j in range (50, 550, 50):
    for i in range(0, 18):
        x_save_load = np.load(
            'I:/Program/Pix2Vox-master/voxel_log/210904_gv_64_spine2_test_epoch_' + str(j) + '/gv_' + str(i).zfill(6) + '.npy')
        imdata = vtk.vtkImageData()
        # this is where the conversion happens
        depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(), deep=3, array_type=vtk.VTK_FLOAT)

        # fill the vtk image data object
        imdata.SetDimensions([64, 64, 64])
        imdata.SetSpacing([1, 1, 1])
        imdata.SetOrigin([0, 0, 0])
        imdata.GetPointData().SetScalars(depthArray)

        writer = vtk.vtkMetaImageWriter()
        writer.SetFileName(
            'I:/Program/Pix2Vox-master/voxel_log/210904_gv_64_spine2_test_epoch_' + str(j) + '_mha/gv_mha_' + str(i).zfill(6) + '.mha')
        writer.SetInputData(imdata)
        writer.Write()

    print(str(j) + ' index ended')
    '''

x_save_load = np.load('I:/DK_Data_Process/i_1-3_Target_Mesh/17_24_spine.npy')
imdata = vtk.vtkImageData()

depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(), deep=3, array_type=vtk.VTK_FLOAT)

# fill the vtk image data object
imdata.SetDimensions([128, 128, 128])
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('I:/DK_Data_Process/i_1-3_Target_Mesh/17_24_spine.mha')
writer.SetInputData(imdata)
writer.Write()

