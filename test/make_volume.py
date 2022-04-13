import numpy as np
import binvox_rw

res = 128

res_volume = np.zeros((res, res, res), dtype=np.uint8)

for i in range(0, res):
    for j in range(0, res):
        for k in range(0, res):
            res_volume[i, j, k] = 56

print(res_volume[32, 32, :])
# np.save('J:/Program/Pix2Vox-master/voxel_log/voxel_process/gv_value_0_2_res_128.npy', res_volume)

voxels = binvox_rw.from_array(res_volume, [res, res, res], [0.0, 0.0, 0.0], 1, fix_coords=True)
with open('J:/Program/Pix2Vox-master/voxel_log/voxel_process/gv_value_56_res_128.binvox', 'wb') as f:
    voxels.write(f)