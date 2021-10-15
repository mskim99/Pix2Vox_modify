import binvox_rw
import numpy as np

with open('J:/DK_Data_Process/i_1-3_Target_Mesh/f_001_vrt_24_res_32_fit.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

data = model.data
pos = np.where(data[:,:,:] == 255)
pos_num = len(pos[0])
print(pos_num)