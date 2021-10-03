import binvox_rw
from vtk.util import numpy_support
import vtk

with open('J:/Program/Pix2Vox-master/voxel_gtv_log/gtv_0000001_a.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

data = model.data

imdata = vtk.vtkImageData()
# this is where the conversion happens
depthArray = numpy_support.numpy_to_vtk(data.ravel(), deep=3, array_type=vtk.VTK_FLOAT)

# fill the vtk image data object
imdata.SetDimensions([32, 32, 32])
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName(
    'J:/Program/Pix2Vox-master/voxel_gtv_log/gtv_0000001_a_bvx.mha')
writer.SetInputData(imdata)
writer.Write()
