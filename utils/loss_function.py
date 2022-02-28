import torch
import utils.network_utils
import math
from torch import autograd
from torch.autograd import Variable

from IQA_pytorch import SSIM

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


def dice_loss(gv, gtv, thres, smooth = 1e-5):
    gv_ref_less = torch.le(gv, thres).float()
    gv_ref_more = torch.le(gv, thres).float()
    gtv_ref_less = torch.le(gtv, thres).float()
    gtv_ref_more = torch.ge(gtv, thres).float()
    FP_loss_intersection = torch.sum(gv_ref_less.mul(gtv_ref_more))
    FP_loss_union = torch.sum(gv_ref_less.add(gtv_ref_more))
    TN_loss_intersection = torch.sum(gv_ref_more.mul(gtv_ref_less))
    TN_loss_union = torch.sum(gv_ref_more.add(gtv_ref_less))
    dice_loss = FP_loss_intersection / (FP_loss_union + smooth) + TN_loss_intersection / (TN_loss_union + smooth)
    return dice_loss


def dice_loss_weight(gv, gtv, thres_min, thres_max, smooth = 1e-5):

    thres = (thres_min + thres_max) / 2.

    gv_ref_less = torch.le(gv, thres).float()
    gv_ref_more = torch.le(gv, thres).float()
    gtv_ref_less = torch.le(gtv, thres).float()
    gtv_ref_more = torch.ge(gtv, thres).float()

    # weighting values
    gv_weight = torch.ones(gv.size())
    gtv_weight = torch.ones(gtv.size())
    gv_weight = utils.network_utils.var_or_cuda(gv_weight)
    gtv_weight = utils.network_utils.var_or_cuda(gtv_weight)

    gv_between_pos = torch.where((gv[:, :, :] <= thres_max) & (gv[:, :, :] >= thres_min))
    gtv_between_pos = torch.where((gtv[:, :, :] <= thres_max) & (gtv[:, :, :] >= thres_min))

    # linear weight
    gv_w = torch.abs((gv[gv_between_pos] - thres) / (thres_max - thres))
    gv_w = gv_w * gv_w
    gtv_w = torch.abs((gtv[gtv_between_pos] - thres) / (thres_max - thres))
    gtv_w = gtv_w * gtv_w
    gv_weight[gv_between_pos] = gv_w
    gtv_weight[gtv_between_pos] = gtv_w

    # tanh weight
    # gv_weight[gv_between_pos] = torch.tanh(10. * torch.abs(gv[gv_between_pos] - thres) / (thres_max - thres))
    # gtv_weight[gtv_between_pos] = torch.tanh(10. * torch.abs(gtv[gtv_between_pos] - thres) / (thres_max - thres))

    gv_ref_less_w = gv_ref_less.mul(gv_weight)
    gv_ref_more_w = gv_ref_more.mul(gv_weight)
    gtv_ref_less_w = gtv_ref_less.mul(gtv_weight)
    gtv_ref_more_w = gtv_ref_more.mul(gtv_weight)

    FP_loss_intersection = torch.sum(gv_ref_less_w.mul(gtv_ref_more_w))
    FP_loss_union = torch.sum(gv_ref_less.add(gtv_ref_more))
    TN_loss_intersection = torch.sum(gv_ref_more.mul(gtv_ref_less))
    TN_loss_union = torch.sum(gv_ref_more_w.add(gtv_ref_less_w))
    dice_loss = FP_loss_intersection / (FP_loss_union + smooth) + TN_loss_intersection / (TN_loss_union + smooth)

    return dice_loss


def ssim_loss_volume(gv, gtv):
    SSIM_loss = SSIM(channels=3)
    loss_total = 0.0
    gv = torch.squeeze(gv)
    gtv = torch.squeeze(gtv)
    for i in range (0, gv.shape[2]):
        gv_part = gv[:, :, i]
        gtv_part = gtv[:, :, i]
        gv_part = gv_part.reshape(1, 1, gv_part.shape[0], gv_part.shape[1])
        gtv_part = gtv_part.reshape(1, 1, gtv_part.shape[0], gtv_part.shape[1])
        ssim_loss_part = SSIM_loss(gv_part, gtv_part, as_loss=False)
        loss_total += ssim_loss_part

    loss_total = loss_total / gv.shape[2]
    return loss_total


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


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1, 1, 1)
    alpha = alpha.expand(real_data.size())

    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.
    return gradient_penalty
