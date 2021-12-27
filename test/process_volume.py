from vtk.util import numpy_support
import vtk
import numpy as np

x_save_load = np.load('J:/Program/Pix2Vox-master/voxel_log/ground_truth_volume/gtv_000000.npy')

# Clip 50%`75~
x_save_load = np.clip(x_save_load, 0.36, 1.0)
x_save_load[np.where(x_save_load[:,:,:] <= 0.36)] = 0.0
np.save('J:/Program/Pix2Vox-master/voxel_log/ground_truth_volume/gtv_000000_low_clip.npy', x_save_load)

imdata = vtk.vtkImageData()

depthArray = numpy_support.numpy_to_vtk(x_save_load.ravel(order='C'), deep=True, array_type=vtk.VTK_FLOAT)

imdata.SetDimensions([128, 128, 128])
# fill the vtk image data object
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('J:/Program/Pix2Vox-master/voxel_log/ground_truth_volume/gtv_000000_low_clip.mha')
writer.SetInputData(imdata)
writer.Write()