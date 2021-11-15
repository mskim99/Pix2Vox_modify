import binvox_rw
from vtk.util import numpy_support
import vtk
import glob

'''
res = 64

for i in range (1, 56):
    image_paths = glob.glob('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_64/gtv_f_' + str(i).zfill(7) + '_a.binvox')
    if len(image_paths) > 0:
        with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_64/gtv_f_' + str(i).zfill(7) + '_a.binvox', 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)

        data = model.data

        imdata = vtk.vtkImageData()
        # this is where the conversion happens
        depthArray = numpy_support.numpy_to_vtk(data.ravel(), deep=3, array_type=vtk.VTK_LONG)

        # fill the vtk image data object
        imdata.SetDimensions([res, res, res])
        imdata.SetSpacing([1, 1, 1])
        imdata.SetOrigin([0, 0, 0])
        imdata.GetPointData().SetScalars(depthArray)

        writer = vtk.vtkMetaImageWriter()
        writer.SetFileName('J:/Program/Pix2Vox-master/voxel_gtv_log/mha/x_64/gtv_f_' + str(i).zfill(7) + '_a.mha')
        writer.SetInputData(imdata)
        writer.Write()
        '''

with open('J:/DK_Data_Process/i_1-3_Target_Mesh/24_spine/res_32_rot_yx_90_y_90_prc_sc_fill/f_001_vrt_24.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

data = model.data

imdata = vtk.vtkImageData()
# this is where the conversion happens
depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=3, array_type=vtk.VTK_INT)

# fill the vtk image data object
imdata.SetDimensions([32, 32, 32])
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('J:/DK_Data_Process/i_1-3_Target_Mesh/24_spine/res_32_rot_yx_90_y_90_prc_sc_fill/f_001_vrt_24.mha')
writer.SetInputData(imdata)
writer.Write()

