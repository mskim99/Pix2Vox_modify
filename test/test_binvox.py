import binvox_rw

with open('J:/Program/Pix2Vox-master/voxel_gtv_log/gtv_0000001_a.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

data = model.data
print(data)