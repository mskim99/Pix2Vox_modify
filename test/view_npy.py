import numpy as np

data = np.load('I:/Program\Pix2Vox-master/voxel_log/210904_gv_64_spine2_test_epoch_500/gv_000005.npy')
'''
for v in np.nditer(data):
    print(v)
    '''
print(data[:, 35, 0, 34])
'''
for i, j in enumerate(data):
    if data[i, j] == 0:
        zero_num = zero_num+1

print(zero_num)
'''