import torch
import utils.network_utils
import math


def loss_gtv(gv, gtv, thres, mv, b=1):

    # Set Weight
    weight = torch.ones(gtv.shape).cuda()
    weight_more = torch.where(gtv[:, :, :] > thres)
    weight_less = torch.where((gtv[:, :, :] < thres) & (gtv[:, :, :] > thres - 0.05))
    weight[weight_more] = mv
    weight[weight_less] = -10.0 * gtv[weight_less] + (10 * thres + 0.5)

    smooth_l1_loss = torch.nn.SmoothL1Loss(reduce=None, reduction='none', beta=b)

    loss_value = smooth_l1_loss(gv, gtv)
    loss_value = weight * loss_value
    loss_value = torch.mean(loss_value)

    return loss_value

def ls_loss(gv, gtv, thres, ep):

    mahv_gv = mahf(gv - 0.5, ep)
    mahv_gv = mahv_gv.cuda()

    # inside_pos = torch.where(gtv[:] >= thres)
    # outside_pos = torch.where(gtv[:] < thres)

    ai_in = cal_bgtm_in(mahv_gv, gtv)
    ai_out = cal_bgtm_out(mahv_gv, gtv)

    gc1 = gtv - ai_in
    gc1 = gc1 * gc1

    gc2 = gtv - ai_out
    gc2 = gc2 * gc2

    gc = gc1 * mahv_gv + gc2 * (1 - mahv_gv)
    loss_value = gc.sum()
    return loss_value

def dice_loss(gv, gtv, thres_min, thres_max, weight, smooth = 1e-5):
    gv_ref = gv
    gtv_ref = gtv
    gv_pos_min = torch.less(gv, thres_min)
    gtv_pos_min = torch.less(gtv, thres_min)
    gv_pos_max = torch.greater(gv, thres_max)
    gtv_pos_max = torch.greater(gtv, thres_max)
    gv_ref[gv_pos_min] = 0.0
    gtv_ref[gtv_pos_min] = 0.0
    gv_ref[gv_pos_max] = 0.0
    gtv_ref[gtv_pos_max] = 0.0
    intersection = torch.sum(gv_ref.mul(gtv_ref))
    union = torch.sum(gv_ref.add(gtv_ref))
    loss_iou = - weight * torch.log((intersection + smooth) / (union + smooth))
    return loss_iou

def cal_bgtm_in(mahv_gv, gtv):
    im_data = torch.zeros(gtv.shape).cuda()
    im_data = gtv * mahv_gv
    im_data = im_data / mahv_gv
    return im_data


def cal_bgtm_out(mahv_gv, gtv):
    im_data = torch.zeros(gtv.shape).cuda()
    im_data = gtv * (1. - mahv_gv)
    im_data = im_data / (1. - mahv_gv)
    return im_data


def mahf(z, ep):
    im_data = torch.zeros(z.shape).cuda()
    im_data[:, :, :] = z[:, :, :] / ep
    im_data = torch.tanh(im_data)
    im_data = 0.5 * (im_data + 1.)
    return im_data

'''
test_gtv = torch.tensor([0.4, 0.5, 0.6, 0.4])
test_gv = torch.tensor([0.4, 0.5, 0.6, 0.4])

ls_loss_value = 100. * ls_loss(test_gv, test_gtv, 0.5, 1.)

print(ls_loss_value.item())
'''