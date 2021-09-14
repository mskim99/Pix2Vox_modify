from vtk.util import numpy_support
import vtk
import numpy as np

x_save_load = np.load('I:\Program/Pix2Vox-master/voxel_log/210904_gv_64_spine2_test_epoch_500/gv_000000.npy')

for k in range(40, 64):
    x_save_load[:, :, :, k] = 0.0

imdata = vtk.vtkImageData()

depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(), deep=3, array_type=vtk.VTK_FLOAT)

imdata.SetDimensions([64, 64, 64])
# fill the vtk image data object
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('I:\Program/Pix2Vox-master/voxel_log/voxel_process/gv_mha_000000_up.mha')
writer.SetInputData(imdata)
writer.Write()