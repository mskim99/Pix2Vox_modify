import torch
from torch.nn import functional as F

class Generator(torch.nn.Module):
    def __init__(self, noise: int = 1000, channel: int = 128):
        super(Generator, self).__init__()
        _c = channel

        self.noise = noise
        self.fc = torch.nn.Linear(1000, 1024 * 8 * 8 * 8)
        self.bn1 = torch.nn.BatchNorm3d(_c * 8)

        self.tp_conv2 = torch.nn.Conv3d(_c * 8, _c * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm3d(_c * 4)

        self.tp_conv3 = torch.nn.Conv3d(_c * 4, _c * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm3d(_c * 2)

        self.tp_conv4 = torch.nn.Conv3d(_c * 2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = torch.nn.BatchNorm3d(_c)

        self.tp_conv5 = torch.nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, noise):
        noise = noise.view(-1, 1000)
        h = self.fc(noise)
        h = h.view(-1, 1024, 8, 8, 8)
        h = F.relu(self.bn1(h))
        # print(h.size())
        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv2(h)
        h = F.relu(self.bn2(h))
        # print(h.size())
        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv3(h)
        h = F.relu(self.bn3(h))
        # print(h.size())
        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv4(h)
        h = F.relu(self.bn4(h))
        # print(h.size())
        h = F.upsample(h, scale_factor=2)
        h = self.tp_conv5(h)
        # print(h.size())
        h = torch.tanh(h)

        return h

