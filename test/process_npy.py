import numpy as np
from vtk.util import numpy_support
import vtk

center_pos = np.array([64, 32, 64])
tumor_size = 24.

data = np.load('J:/Program/Pix2Vox-master/voxel_log/211221_3_loss_GAN_test_epoch_400_G_1_D_linear_increase_10_r_img_4_L1_4_SSIM_2_IoU_drp_g_e_0_375_lr_1e-4_norm_res_128/gv/gv_000007.npy')

for x in range (0, 128):
    for y in range (0, 128):
        for z in range (0, 128):

            subs = np.array(center_pos - [x, y, z]).astype(float)
            dist = np.linalg.norm(subs)

            if dist < tumor_size:
                data[x, y, z] = 1.0

np.save('J:/DK_Data_Process/Add_Tumor/gv_000007_add_tumor_24.npy', data)

imdata = vtk.vtkImageData()

depthArray = numpy_support.numpy_to_vtk(data.ravel(order='C'), deep=True, array_type=vtk.VTK_FLOAT)

imdata.SetDimensions([128, 128, 128])
# fill the vtk image data object
imdata.SetSpacing([1, 1, 1])
imdata.SetOrigin([0, 0, 0])
imdata.GetPointData().SetScalars(depthArray)

writer = vtk.vtkMetaImageWriter()
writer.SetFileName('J:/DK_Data_Process/Add_Tumor/gv_000007_add_tumor_24.mha')
writer.SetInputData(imdata)
writer.Write()