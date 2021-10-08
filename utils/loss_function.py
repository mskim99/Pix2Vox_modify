import torch
import utils.network_utils


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