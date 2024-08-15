from .memory import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def pad(x, ref=None, h=None, w=None):
    assert not (ref is None and h is None and w is None)
    _, _, h1, w1 = x.shape
    if not ref is None:
        _, _, h2, w2 = ref.shape
    else:
        h2, w2 = h, w
    if not h1 == h2 or not w1 == w2:
        x = F.pad(x, (0, w2 - w1, 0, h2 - h1), mode='replicate')
    return x


class MASK_GENE(nn.Module):
    def __init__(self, io_ch):
        super(MASK_GENE, self).__init__()
        self.mask_conv0 = nn.Conv2d(io_ch * 2 + 1, io_ch * 2, 1)
        self.mask_ac = nn.ReLU(inplace=True)
        self.mask_conv1 = nn.Conv2d(io_ch * 2, io_ch * 2, 1)
        self.io_ch = io_ch

    def forward(self, x):
        # x = torch.cat([x2, x1], dim=1)
        mask = F.sigmoid(self.mask_conv1(self.mask_ac(self.mask_conv0(x))))
        # mask_removal, mask_inpainting = torch.split(mask, self.io_ch, dim=1)
        return mask


# 合并卷积、归一化与激活函数
class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, norm=None,
                 num_groups=8, act='relu', negative_slope=0.1, inplace=False, reflect=False):
        super(ConvNormAct, self).__init__()
        self.layer = nn.Sequential()
        if reflect:  # 反射填充
            self.layer.add_module('pad', nn.ReflectionPad2d(padding))
            self.layer.add_module('conv',
                                  nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=0, dilation=dilation, bias=bias))
        else:
            self.layer.add_module('conv',
                                  nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, bias=bias))
        if norm == 'bn':
            self.layer.add_module('norm', nn.BatchNorm2d(num_features=out_channels))
        elif norm == 'gn':
            self.layer.add_module('norm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
        else:
            pass
        if act == 'relu':
            self.layer.add_module('act', nn.ReLU(inplace=inplace))
        if act == 'relu6':
            self.layer.add_module('act', nn.ReLU6(inplace=inplace))
        elif act == 'elu':
            self.layer.add_module('act', nn.ELU(alpha=1.0))
        elif act == 'lrelu':
            self.layer.add_module('act', nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace))
        elif act == 'sigmoid':
            self.layer.add_module('act', nn.Sigmoid())
        elif act == 'tanh':
            self.layer.add_module('tanh', nn.Tanh())
        else:
            pass

    def forward(self, x):
        y = self.layer(x)
        return y


# 门控卷积
class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, norm=None,
                 num_groups=8, act='elu', negative_slope=0.1, inplace=True, full=True, reflect=True):
        super(GatedConv, self).__init__()
        self.conv = ConvNormAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, bias=bias,
                                norm=norm, num_groups=num_groups, act=act, negative_slope=negative_slope,
                                reflect=reflect)
        if full:
            self.gate = ConvNormAct(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias, act='sigmoid',
                                    reflect=reflect)
        else:
            self.gate = ConvNormAct(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, bias=bias, act='sigmoid', norm=None,
                                    num_groups=1, reflect=reflect)

    def forward(self, x):
        return self.conv(x) * self.gate(x)


# 动态调整输入张量的权重
class DELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(DELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel * 3, channel * 3 // reduction),
            nn.ELU(inplace=True),
            nn.Linear(channel * 3 // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            _mean = x.mean(dim=[2, 3])
            _std = x.std(dim=[2, 3])
            _max = x.max(dim=2)[0].max(dim=2)[0]
        feat = torch.cat([_mean, _std, _max], dim=1)
        b, c, _, _ = x.shape
        y = self.fc(feat).view(b, c, 1, 1)
        return x * y


# 残差块
class ResBlock(nn.Module):
    def __init__(self, channels, blocks=3, resscale=0.1, kernel_size=0, dilations=[2], gatedconv=False, enhance='de',
                 ppm=True):
        super(ResBlock, self).__init__()
        # self.preconv = ConvNormAct(3, 64, kernel_size=3, padding=1)
        self.convs = nn.ModuleList(
            [self._build_layer(channels, kernel_size, dilations, gatedconv, enhance) for i in range(blocks)]
        )
        self.resscale = resscale
        self.ppm = PyramidPooling(channels, channels, ct_channels=channels) if ppm else None
        # self.outconv = ConvNormAct(64, 3, kernel_size=3, padding=1)

    def _build_layer(self, channels, kernel_size=0, dilations=[2], gatedconv=False, enhance='de'):
        conv = GatedConv if gatedconv else ConvNormAct
        layer = nn.Sequential(
            conv(channels, channels, kernel_size=kernel_size, padding=(kernel_size // 2))
            # DilationBlock(channels, channels, channels, dilations=dilations, gatedconv=gatedconv)
        )

        if enhance == 'de':
            layer.add_module('de', DELayer(channels))
        return layer

    def forward(self, x):
        # x = self.preconv(x)
        for conv in self.convs:
            x = conv(x) + x * self.resscale
        if not self.ppm is None:
            x = self.ppm(x)
        return x


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat(
            [F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


# 下采样块
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernels=None):
        super(DownBlock, self).__init__()
        assert isinstance(kernels, list) or isinstance(kernels, tuple) or isinstance(kernels, int)
        Conv = ConvNormAct
        if isinstance(kernels, int):
            assert mid_channels is None
            self.conv = Conv(in_channels, out_channels, kernel_size=kernels, stride=2, padding=kernels // 2)
        else:
            if mid_channels is None:
                mid_channels = out_channels
            i_channels = [in_channels] + [mid_channels] * (len(kernels) - 1)
            o_channels = [mid_channels] * (len(kernels) - 1) + [out_channels]
            conv = [Conv(i_channels[0], o_channels[0], kernel_size=kernels[0], stride=2, padding=kernels[0] // 2)]
            for i in range(1, len(kernels)):
                conv.append(Conv(i_channels[i], o_channels[i], kernel_size=kernels[i], padding=kernels[i] // 2))
            self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


# 感受野扩张，连续的两层卷积进行感受野扩张
class DilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, dilations):
        super(DilationBlock, self).__init__()
        assert isinstance(dilations, list) or isinstance(dilations, tuple)
        i_channels = [in_channels] + [mid_channels] * (len(dilations) - 1)
        o_channels = [mid_channels] * len(dilations)
        Conv = ConvNormAct
        self.conv = nn.Sequential(
            *[Conv(i_channels[i], o_channels[i], kernel_size=3, padding=dilations[i], dilation=dilations[i]) \
              for i in range(len(dilations))]
        )
        self.out = Conv(in_channels + mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        conv = self.conv(x)
        out = self.out(torch.cat([x, conv], dim=1))
        return out


# bilinear为True时，双线性插值
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, add_channels=None, kernels=None,
                 bilinear=False, shape=None):
        super(UpBlock, self).__init__()
        assert isinstance(kernels, list) or isinstance(kernels, tuple)
        Conv = ConvNormAct
        if mid_channels is None:
            mid_channels = out_channels
        if isinstance(mid_channels, int):
            i_channels = [in_channels] + [mid_channels] * (len(kernels) - 1)
            o_channels = [mid_channels] * (len(kernels) - 1) + [out_channels]
        else:
            assert isinstance(mid_channels, list) or isinstance(mid_channels, tuple)
            assert len(mid_channels) == len(kernels) - 1
            i_channels = [in_channels] + list(mid_channels)
            o_channels = list(mid_channels) + [out_channels]
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if not add_channels is None:
                i_channels[0] = i_channels[0] + add_channels
            conv = []
            for i in range(len(kernels)):
                conv.append(Conv(i_channels[i], o_channels[i], kernel_size=kernels[i], padding=kernels[i] // 2))
            self.conv = nn.Sequential(*conv)
        else:
            self.up = nn.ConvTranspose2d(i_channels[0], o_channels[0], kernel_size=kernels[0], stride=2,
                                         padding=kernels[0] // 2, output_padding=1)
            if not add_channels is None:
                i_channels[1] = i_channels[1] + add_channels
            conv = []
            for i in range(1, len(kernels)):
                conv.append(Conv(i_channels[i], o_channels[i], kernel_size=kernels[i], padding=kernels[i] // 2))
            self.conv = nn.Sequential(*conv)

    def forward(self, x, feat=None, shape=None):
        assert not feat is None or not shape is None
        up = self.up(x)
        up = pad(up, ref=feat, h=None if shape is None else shape[0], w=None if shape is None else shape[1])
        if not feat is None:
            return self.conv(torch.cat([up, feat], dim=1))
        else:
            return self.conv(up)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, temp_channels=512):
        super(Encoder, self).__init__()
        self.encoder_conv1 = nn.Sequential(
            ConvNormAct(in_channels, 64, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(64, 64, kernel_size=3, padding=1, norm='bn', act='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder_conv2 = nn.Sequential(
            ConvNormAct(64, 128, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(128, 128, kernel_size=3, padding=1, norm='bn', act='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder_conv3 = nn.Sequential(
            ConvNormAct(128, 256, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(256, 256, kernel_size=3, padding=1, norm='bn', act='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder_conv4 = nn.Sequential(
            ConvNormAct(256, 512, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(512, 512, kernel_size=3, padding=1, norm=None, act=None)
        )

    def forward(self, x):
        conv1 = self.encoder_conv1(x)
        conv2 = self.encoder_conv2(conv1)
        conv3 = self.encoder_conv3(conv2)
        conv4 = self.encoder_conv4(conv3)
        return conv4


class Decoder(nn.Module):
    def __init__(self, temp_channels=1024, out_channels=3):
        super(Decoder, self).__init__()

        def upsample(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)
            )

        self.decoder_conv1 = nn.Sequential(
            ConvNormAct(1024, 512, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(512, 512, kernel_size=3, padding=1, norm='bn', act='relu')
        )
        self.up1 = upsample(512, 512)
        self.decoder_conv2 = nn.Sequential(
            ConvNormAct(512, 256, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(256, 256, kernel_size=3, padding=1, norm='bn', act='relu')
        )
        self.up2 = upsample(256, 256)
        self.decoder_conv3 = nn.Sequential(
            ConvNormAct(256, 128, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(128, 128, kernel_size=3, padding=1, norm='bn', act='relu')
        )
        self.up3 = upsample(128, 128)
        self.decoder_conv4 = nn.Sequential(
            ConvNormAct(128, 64, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(64, 64, kernel_size=3, padding=1, norm='bn', act='relu')
        )
        self.final_conv = ConvNormAct(64, 3, kernel_size=3, padding=1, norm=None, act='tanh')

    def forward(self, fea):
        conv1 = self.decoder_conv1(fea)
        up1 = self.up1(conv1)

        conv2 = self.decoder_conv2(up1)
        up2 = self.up2(conv2)

        conv3 = self.decoder_conv3(up2)
        up3 = self.up3(conv3)

        conv4 = self.decoder_conv4(up3)
        output = self.final_conv(conv4)

        return output


class WeightDistributionModule(nn.Module):
    # 输入removal和inpainting，输入通道应该为3，
    def __init__(self, in_ch=3, out_ch=3):
        super(WeightDistributionModule, self).__init__()

        self.mask_gene = MASK_GENE(in_ch)

        # partial conv
        self.cv = nn.Conv2d(in_ch * 2, in_ch * 2, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch * 2, track_running_stats=False)
        self.ac = nn.ReLU(inplace=True)
        self.pooling = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.out = ConvNormAct(in_ch * 2, 3, kernel_size=3, padding=1)

    def forward(self, removal, inpainting, detect):
        # pdb.set_trace()
        x1 = torch.cat([removal, inpainting, detect], dim=1)
        x = torch.cat([removal, inpainting], dim=1)
        mask = self.mask_gene(x1)
        x = self.cv(x * mask)
        mask_avg = torch.mean(self.pooling(mask), dim=1, keepdim=True)
        mask_avg[mask_avg == 0] = 1
        x = x * (1 / mask_avg)
        x = self.bn(x)
        x = self.ac(x)
        x = self.out(x)

        return x, mask


class MTNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(MTNet, self).__init__()
        self.encoder_conv1 = nn.Sequential(
            ConvNormAct(in_channels, 64, kernel_size=7, padding=3),
            ConvNormAct(64, 64, kernel_size=7, padding=3),
            ConvNormAct(64, 64, kernel_size=7, padding=3)
        )
        self.encoder_conv2 = nn.Sequential(
            DownBlock(64, 128, kernels=[3, 3]),
            DilationBlock(128, 128, 128, dilations=[2, 4])
        )
        self.encoder_conv3 = nn.Sequential(
            DownBlock(128, 256, kernels=[3, 3]),
            DilationBlock(256, 256, 256, dilations=[2, 4])
        )
        self.resblock = ResBlock(256, blocks=3, resscale=0.1, kernel_size=3, gatedconv=True, dilations=[2, 4])
        self.up1 = UpBlock(256, 128, 128, add_channels=128, kernels=[5, 5], bilinear=True)
        self.up2 = UpBlock(128, 64, 64, add_channels=64, kernels=[5, 5], bilinear=True)
        # detection branch
        self.detect_branch = nn.Sequential(
            ConvNormAct(64 + 3, 64, kernel_size=3, padding=1),
            ConvNormAct(64, 64, kernel_size=3, padding=1),
            ConvNormAct(64, 1, kernel_size=3, padding=1, norm=None, act='sigmoid')
        )
        self.elimination_branch = nn.Sequential(
            ConvNormAct(64 + 4, 64, kernel_size=3, padding=1),
            ConvNormAct(64, 64, kernel_size=3, padding=1),
            ConvNormAct(64, 3, kernel_size=3, padding=1, norm=None)
        )

    def forward(self, x):
        conv1 = self.encoder_conv1(x)
        conv2 = self.encoder_conv2(conv1)
        conv3 = self.encoder_conv3(conv2)
        conv3 = self.resblock(conv3)
        up1 = self.up1(conv3, conv2)
        up2 = self.up2(up1, conv1)
        detect = self.detect_branch(torch.cat([x, up2], dim=1))
        eliminate = self.elimination_branch(torch.cat([x, up2, detect], dim=1))
        return detect, eliminate, up2


class ERRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ERRNet, self).__init__()
        self.mtnet = MTNet()
        self.encoder = Encoder(3, 512)
        self.decoder2 = Decoder(1024, 3)
        # weight distribution module
        self.WDM = WeightDistributionModule()
        # memory block(read and update)
        self.memory = Memory(memory_size=512, feature_dim=512, key_dim=512, temp_update=0.1, temp_gather=0.1)

        self.refined_removal = nn.Sequential(
            GatedConv(64 + 7, 64, kernel_size=7, padding=3),
            ResBlock(64, blocks=3, resscale=1.0, kernel_size=3, gatedconv=True, dilations=[2, 4]),
            ConvNormAct(64, 3, kernel_size=3, padding=1, norm=None, act='relu6')
        )

    def forward(self, x, keys, train):
        detect, eliminate, up = self.mtnet(x)

        fea = self.encoder(x)
        # resize the detect result to cat with the fea
        re_fea = fea

        # content inpainting branch(memory block)
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(
                re_fea, keys, train)
            inpainting = self.decoder2(updated_fea)
        else:
            spreading_loss = None
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss = self.memory(re_fea, keys,
                                                                                                       train)
            inpainting = self.decoder2(updated_fea)

        removal, weight_mask = self.WDM(eliminate, inpainting, detect)

        result = self.refined_removal(torch.cat([x, eliminate, up, detect], dim=1))

        return {'result': result, 'detect': detect.squeeze(1), 'removal': eliminate, 'inpainting': inpainting,
                'weight_mask': weight_mask,
                'm_items': keys, 'separateness_loss': gathering_loss, 'compactness_loss': spreading_loss}


class FlowNet(nn.Module):
    def __init__(self):
        super(FlowNet, self).__init__()
        self.conv = nn.Sequential(
            ConvNormAct(6, 64, kernel_size=3, padding=1, stride=2, act='lrelu'),
            ConvNormAct(64, 128, kernel_size=3, padding=1, stride=2, norm='bn', act='lrelu'),
            ConvNormAct(128, 256, kernel_size=3, padding=1, stride=2, norm='bn', act='lrelu'),
            ConvNormAct(256, 512, kernel_size=3, padding=1, stride=2, norm='bn', act='lrelu'),
            ConvNormAct(512, 1024, kernel_size=3, padding=1, stride=2, norm='bn', act='lrelu'),
        )
        self.transconv = nn.Sequential(
            ConvNormAct(1024, 512, kernel_size=3, padding=1, stride=2, norm='bn', act='relu', reflect=False),
            ConvNormAct(512, 256, kernel_size=3, padding=1, stride=2, norm='bn', act='relu', reflect=False),
            ConvNormAct(256, 128, kernel_size=3, padding=1, stride=2, norm='bn', act='relu', reflect=False),
            ConvNormAct(128, 64, kernel_size=3, padding=1, stride=2, norm='bn', act='relu', reflect=False),
        )
        self.out = ConvNormAct(64, 2, kernel_size=3, padding=1, stride=2, act='tanh', reflect=False)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.transconv(x)
        x = self.out(x)

        return x
