import meshio

mesh = meshio.read('I:\Program/Pix2Vox-master/voxel_log/voxel_process/gv_mha_000000_up.vtu')

mesh.write("I:\Program/Pix2Vox-master/voxel_log/voxel_process/gv_mha_000000_up.obj")