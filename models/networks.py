import torch
import torch.nn as nn
from torch.nn import init


def weights_init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='edsr'):
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'edsr':
        pass
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    print('The size of receptive field: %d' % receptive_field(net))


def receptive_field(net):
    def _f(output_size, ksize, stride, dilation):
        return (output_size - 1) * stride + ksize * dilation - dilation + 1

    stats = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            stats.append((m.kernel_size, m.stride, m.dilation))

    rsize = 1
    for (ksize, stride, dilation) in reversed(stats):
        if type(ksize) == tuple:
            ksize = ksize[0]
        if type(stride) == tuple:
            stride = stride[0]
        if type(dilation) == tuple:
            dilation = dilation[0]
        rsize = _f(rsize, ksize, stride, dilation)
    return rsize


def define_D(opt, in_channels=3):
    use_sigmoid = False

    netD = Discriminator_VGG(in_channels, use_sigmoid=use_sigmoid)
    init_weights(netD, init_type='kaiming')

    if len(opt.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda(opt.gpu_ids[0])

    return netD


class Discriminator_VGG(nn.Module):
    def __init__(self, in_channels=3, use_sigmoid=True):
        super(Discriminator_VGG, self).__init__()

        def conv(*args, **kwargs):
            return nn.Conv2d(*args, **kwargs)

        num_groups = 32

        body = [
            conv(in_channels, 64, kernel_size=3, padding=1),  # 224
            nn.LeakyReLU(0.2),

            conv(64, 64, kernel_size=3, stride=2, padding=1),  # 112
            nn.GroupNorm(num_groups, 64),
            nn.LeakyReLU(0.2),

            conv(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 128),
            nn.LeakyReLU(0.2),

            conv(128, 128, kernel_size=3, stride=2, padding=1),  # 56
            nn.GroupNorm(num_groups, 128),
            nn.LeakyReLU(0.2),

            conv(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 256),
            nn.LeakyReLU(0.2),

            conv(256, 256, kernel_size=3, stride=2, padding=1),  # 28
            nn.GroupNorm(num_groups, 256),
            nn.LeakyReLU(0.2),

            conv(256, 512, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),

            conv(512, 512, kernel_size=3, stride=2, padding=1),  # 14
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),

            conv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),

            conv(512, 512, kernel_size=3, stride=2, padding=1),  # 7
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),
        ]

        tail = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        ]

        if use_sigmoid:
            tail.append(nn.Sigmoid())

        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.body(x)
        out = self.tail(x)
        return out
