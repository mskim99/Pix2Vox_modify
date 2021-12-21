# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch
import joblib

from datetime import datetime as dt


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def save_checkpoints(cfg, file_path, epoch_idx, generator, generator_solver, discriminator, discriminator_solver, volume_scaler, image_scaler):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'generator_state_dict': generator.state_dict(),
        'encoder_solver_state_dict': generator_solver.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'discriminator_solver_state_dict': discriminator_solver.state_dict()
    }

    torch.save(checkpoint, file_path)
    joblib.dump(volume_scaler, '/home/jzw/work/pix2vox/output/logs/checkpoints2/volume_scaler.pkl')
    joblib.dump(image_scaler, '/home/jzw/work/pix2vox/output/logs/checkpoints2/image_scaler.pkl')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
