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

# with open('J:/DK_Data_Process/i_1-3_Target_Mesh/23_24_spine/res_128_fit/f_001_vrt_23_24_res_128.binvox', 'rb') as f:
with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_256/gtv_f_0000026_a.binvox', 'rb') as f:
# with open('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc256/KISTI_Vox/00000024/f_0000001/model.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

data = model.data

imdata = vtk.vtkImageData()
# this is where the conversion happens
depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=3, array_type=vtk.VTK_INT)

# fill the vtk image data object
imdata.SetDimensions([256, 256, 256])
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
# writer.SetFileName('J:/DK_Data_Process/i_1-3_Target_Mesh/23_24_spine/res_128_fit/f_001_vrt_23_24_res_128.mha')
writer.SetFileName('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x_256/gtv_f_0000026_a.mha')
# writer.SetFileName('J:/Program/Pix2Vox-master/Pix2Vox-master/datasets/KISTI_volume_CT_mesh_sc256/KISTI_Vox/00000024/f_0000001/model.mha')
writer.SetInputData(imdata)
writer.Write()

