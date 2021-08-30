from vtk.util import numpy_support
import vtk
import numpy as np

for i in range (0, 18):
    x_save_load = np.load('I:/Program/Pix2Vox-master/voxel_log/210829_gv_xray_test_epoch_250/gv_' + str(i).zfill(6) + '.npy')
    imdata = vtk.vtkImageData()
    # this is where the conversion happens
    depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(), deep=3, array_type=vtk.VTK_FLOAT)

    # fill the vtk image data object
    imdata.SetDimensions([32, 32, 32])
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(depthArray)

    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName('I:/Program/Pix2Vox-master/voxel_log/210829_gv_xray_test_epoch_250_mha/gv_mha_' + str(i).zfill(6) +'.mha')
    writer.SetInputData(imdata)
    writer.Write()

    if i % 100 == 0:
        print(str(i) + ' file ended')
