import binvox_rw
from vtk.util import numpy_support
import vtk
import glob

for i in range (1, 2):
    image_paths = glob.glob('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2_thres_0_35/gtv_f_001_a.binvox')
    if len(image_paths) > 0:
        with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2_thres_0_35/gtv_f_001_a.binvox', 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)

        data = model.data

        imdata = vtk.vtkImageData()
        # this is where the conversion happens
        depthArray = numpy_support.numpy_to_vtk(data.ravel(), deep=3, array_type=vtk.VTK_LONG)

        # fill the vtk image data object
        imdata.SetDimensions([32, 32, 32])
        imdata.SetSpacing([1, 1, 1])
        imdata.SetOrigin([0, 0, 0])
        imdata.GetPointData().SetScalars(depthArray)

        writer = vtk.vtkMetaImageWriter()
        writer.SetFileName('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2_thres_0_35/gtv_f_001_a.mha')
        writer.SetInputData(imdata)
        writer.Write()

'''
with open('J:/DK_Data_Process/i_1-3_Target_Mesh/24_spine_f1_rot.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

data = model.data

imdata = vtk.vtkImageData()
# this is where the conversion happens
depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=3, array_type=vtk.VTK_FLOAT)

# fill the vtk image data object
imdata.SetDimensions([32, 32, 32])
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('J:/DK_Data_Process/i_1-3_Target_Mesh/24_spine_f1_rot.mha')
writer.SetInputData(imdata)
writer.Write()
'''
