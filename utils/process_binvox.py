import glob
import binvox_rw
import numpy as np
import math

orig_z = 15
for i in range(1, 58):
    image_paths = glob.glob('J:/DK_Data_Process/i_1-3_Target_Mesh/24_spine/res_32_rot_xz_90_y_180/m_' + str(i).zfill(3) + '_vrt_24.binvox')
    if len(image_paths) > 0:
        with open('J:/DK_Data_Process/i_1-3_Target_Mesh/24_spine/res_32_rot_xz_90_y_180/m_' + str(i).zfill(3) + '_vrt_24.binvox', 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)
        data = model.data
        res_volume = np.zeros((32, 32, 32), dtype=np.uint8)

        # for j in range(0, 32):
            # res_volume[:, :, j] = data[:, :, math.ceil(float(j) * float(orig_z) / 32.)] * 255

        res_volume[:, :, :] = data[:, :, :] * 255.

        voxels = binvox_rw.from_array(res_volume, [32, 32, 32], [0.0, 0.0, 0.0], 1, fix_coords=True)
        with open('J:/DK_Data_Process/i_1-3_Target_Mesh/24_spine/res_32_rot_xz_90_y_180/m_' + str(i).zfill(3) + '_vrt_24_prc.binvox', 'wb') as f:
            voxels.write(f)

        print(str(i) + ' finished')