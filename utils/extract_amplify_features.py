import numpy as np
import torch
# import glob
# import binvox_rw
# import binvox_visualization_offline as bvo

def Extract_Amplify_Features(volume, thres, res):

    res_volume = torch.zeros([1, res, res, res])

    # Thresholding
    remain_pos = np.where(volume[:, :, :, :] >= thres)
    remove_pos = np.where(volume[:, :, :, :] < thres)
    res_volume[remain_pos] = 1.
    res_volume[remove_pos] = 0

    return res_volume


def Extract_Amplify_Features_grad_m(volume, thres_min, thres_max, res, mult):

    res_volume = torch.zeros([1, res, res, res])
    volume = torch.tensor(volume)

    # Thresholding
    over_pos = np.where(volume[:, :, :, :] >= thres_max)
    between_pos = np.where((volume[:, :, :, :] < thres_max) & (volume[:, :, :, :] >= thres_min))
    less_pos = np.where(volume[:, :, :, :] < thres_min)
    res_volume[over_pos] = 1.
    res_volume[between_pos] = (volume[between_pos] - thres_min) / (thres_max - thres_min)
    res_volume[less_pos] = 0.

    saturate_pos = np.where(res_volume[:, :, :, :] > 1.)
    res_volume[saturate_pos] = 1.

    degenerate_pos = np.where(res_volume[:, :, :, :] < 0.)
    res_volume[degenerate_pos] = 0.

    return res_volume


def Extract_Amplify_Features_grad(volume, thres, res, mult):

    res_volume = torch.zeros([1, res, res, res])
    volume = torch.tensor(volume)

    # Thresholding
    remain_pos = np.where(volume[:, :, :, :] >= thres)
    remove_pos = np.where(volume[:, :, :, :] < thres)
    res_volume[remain_pos] = volume[remain_pos] * mult
    res_volume[remove_pos] = 0

    saturate_pos = np.where(res_volume[:, :, :, :] > 1.)
    res_volume[saturate_pos] = 1.

    return res_volume

'''
image_paths = glob.glob('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2/gtv_f_0000001_a.binvox')
if len(image_paths) > 0:
    with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2/gtv_f_0000001_a.binvox', 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    volume = model.data
    bvo.binvox_visualize_png(volume, 'J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2/gtv_f_0000001_a.png', 178)
    processed_volume = Extract_Amplify_Features_grad(volume, 89, 32, 1.5)
    bvo.binvox_visualize_png(processed_volume, 'J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2/gtv_f_0000001_a_cv.png', 178)

    voxels = binvox_rw.from_array(processed_volume, [32, 32, 32], [0.0, 0.0, 0.0], 1, fix_coords=True)
    with open('J:/Program/Pix2Vox-master/voxel_gtv_log/binvox/x2/gtv_f_0000001_a_cv.binvox',
              'wb') as f:
        voxels.write(f)
        '''


