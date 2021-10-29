import numpy as np
import torch

def Extract_Amplify_Features(volume, thres, res):

    res_volume = torch.zeros([1, res, res, res])

    # Thresholding
    remain_pos = np.where(volume[:, :, :, :] >= thres)
    remove_pos = np.where(volume[:, :, :, :] < thres)
    res_volume[remain_pos] = 1.
    res_volume[remove_pos] = 0

    return res_volume
'''
image_paths = glob.glob('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2/gtv_f_0000001_a.binvox')
if len(image_paths) > 0:
    with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2/gtv_f_0000001_a.binvox', 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    volume = model.data
    bvo.binvox_visualize_png(volume, 'J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2/gtv_f_0000001_a.png', 255)
    processed_volume = Extract_Amplify_Features(volume, 89, 32)
    bvo.binvox_visualize_png(processed_volume, 'J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2/gtv_f_0000001_a_cv.png', 255)
    '''

