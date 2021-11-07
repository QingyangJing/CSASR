# modified from: https://github.com/AbdulMoqeet/MAFFSRN

import math
from argparse import Namespace
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from collections import OrderedDict
from models import MPNCOV


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    # p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    # return sequential(p, c, n, a)
    return sequential(c, n, a)


def norm(norm_type, out_nc):
    norm_type = norm_type.lower()
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(out_nc)
    elif norm_type == 'in':
        layer = nn.InstanceNorm2d(out_nc)
    elif norm_type == 'gn':
        layer = nn.GroupNorm(4, out_nc)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(norm_type))
    return layer


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Tail(nn.Module):
    def __init__(self, args, scale, n_feats, kernel_size, wn):
        super(Tail, self).__init__()
        out_feats = scale * scale * args.n_colors
        # self.tail_k3 = wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2, dilation=1))
        # self.tail_k5 = wn(nn.Conv2d(n_feats, out_feats, 5, padding=5//2, dilation=1))
        self.tail_k3 = nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2, dilation=1)
        self.tail_k5 = nn.Conv2d(n_feats, out_feats, 5, padding=5 // 2, dilation=1)
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.scale_k3 = Scale(0.5)
        self.scale_k5 = Scale(0.5)
        self.conv = nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=1)

    def forward(self, x):
        x0 = self.pixelshuffle(self.scale_k3(self.tail_k3(x)))
        x1 = self.pixelshuffle(self.scale_k5(self.tail_k5(x)))

        return x0 + x1


def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class FFB(nn.Module):
    def __init__(self, n_feats, wn, act=nn.ReLU(True)):
        super(FFB, self).__init__()

        self.b0 = CSA(n_feats=n_feats, reduction_factor=4)
        self.b1 = CSA(n_feats=n_feats, reduction_factor=4)
        self.b2 = CSA(n_feats=n_feats, reduction_factor=4)
        self.b3 = CSA(n_feats=n_feats, reduction_factor=4)

        # self.reduction1 = wn(nn.Conv2d(n_feats*2, n_feats, 1))
        # self.reduction2 = wn(nn.Conv2d(n_feats*2, n_feats, 1))
        # self.reduction3 = wn(nn.Conv2d(n_feats*2, n_feats, 1))
        self.reduction1 = nn.Conv2d(n_feats * 2, n_feats, 1)
        self.reduction2 = nn.Conv2d(n_feats * 2, n_feats, 1)
        self.reduction3 = nn.Conv2d(n_feats * 2, n_feats, 1)
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0) + x0
        x2 = self.b2(x1) + x1
        x3 = self.b3(x2)

        res1 = self.reduction1(channel_shuffle(torch.cat([x0, x3], dim=1), 2))
        res2 = self.reduction2(channel_shuffle(torch.cat([x1, x2], dim=1), 2))
        res = self.reduction3(channel_shuffle(torch.cat([res1, res2], dim=1), 2))

        return self.res_scale(res) + self.x_scale(x)


# second-order Channel attention (SOCA)
class SOCA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SOCA, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        # subsample
        # subsample_scale = 2
        # subsample = nn.Upsample(size=(h // subsample_scale, w // subsample_scale), mode='nearest')
        # x_sub = subsample(x)
        # max_pool = nn.MaxPool2d(kernel_size=2)
        # max_pool = nn.AvgPool2d(kernel_size=2)
        # x_sub = self.max_pool(x)
        ##
        ## MPN-COV
        cov_mat = MPNCOV.CovpoolLayer(x_sub)  # Global Covariance pooling layer
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat,
                                         5)  # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt, 1)
        cov_mat_sum = cov_mat_sum.view(batch_size, C, 1, 1)
        # y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_cov = self.conv_du(cov_mat_sum)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_cov * x


class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ESA(nn.Module):
    def __init__(self, n_feats, reduction_factor=4, distillation_rate=0.25):
        super(ESA, self).__init__()
        self.reduce_channels = nn.Conv2d(n_feats, n_feats // reduction_factor, 1)
        self.reduce_spatial_size = nn.Conv2d(n_feats // reduction_factor, n_feats // reduction_factor, 3, stride=2,
                                             padding=1)
        self.pool = nn.MaxPool2d(7, stride=3)
        self.increase_channels = conv_block(n_feats // reduction_factor, n_feats, 1)

        self.conv1 = conv_block(n_feats // reduction_factor, n_feats // reduction_factor, 3, dilation=1,
                                act_type='lrelu')
        self.conv2 = conv_block(n_feats // reduction_factor, n_feats // reduction_factor, 3, dilation=2,
                                act_type='lrelu')

        self.sigmoid = nn.Sigmoid()

        self.bottom11 = conv_block(n_feats, n_feats, 1, act_type=None)
        self.bottom11_dw = conv_block(n_feats, n_feats, 5, groups=n_feats, act_type=None)

    def forward(self, x):
        rc = self.reduce_channels(x)
        rs = self.reduce_spatial_size(rc)
        pool = self.pool(rs)
        # conv = self.conv2(pool)
        # conv = conv + self.conv1(pool)
        conv = self.conv2(pool)
        conv = self.conv1(conv)
        up = torch.nn.functional.upsample(conv, size=(rc.shape[2], rc.shape[3]), mode='nearest')
        up = up + rc
        # out = (self.sigmoid(self.increase_channels(up)) * x) * self.sigmoid(self.bottom11_dw(self.bottom11(x)))
        out = (self.sigmoid(self.increase_channels(up)) * x)
        return out


class CSA(nn.Module):
    def __init__(self, n_feats, reduction_factor=4, distillation_rate=0.25):
        super(CSA, self).__init__()

        self.conv00 = conv_block(n_feats, n_feats, 3, act_type=None)
        self.conv01 = conv_block(n_feats, n_feats, 3, act_type='lrelu')

        self.eca = ECA(n_feats)
        self.esa = ESA(n_feats, reduction_factor)

    def forward(self, x):
        x = self.conv00(self.conv01(x))
        eca_ = self.eca(x)
        esa_ = self.esa(eca_)
        return esa_


class CSASR(nn.Module):
    def __init__(self, args):
        super(CSASR, self).__init__()
        self.args = args

        # hyper-params
        self.scale = args.scale
        n_FFBs = args.n_FFBs
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.LeakyReLU(True)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        # self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
        #     [0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])

        # define head module
        head = []
        # head.append(wn(nn.Conv2d(3, n_feats, 3, padding=3//2)))
        head.append(nn.Conv2d(3, n_feats, 3, padding=3 // 2))

        # define body module
        body = []
        for i in range(n_FFBs):
            body.append(
                FFB(n_feats, wn=wn, act=act))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            self.tail = Tail(args, self.scale, n_feats, kernel_size, wn)

    def forward(self, x):
        input = x
        x = self.head(x)
        x = self.body(x)
        if self.args.no_upsampling:
            return x
        else:
            x = self.tail(x)
            return x + torch.nn.functional.upsample(input, scale_factor=self.scale, mode='bicubic')

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 or name.find('skip') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


def make_csasr(args_use, scale=4, n_FFBs=4, n_feats=32, no_upsampling=False):
    args = Namespace()
    args.scale = scale
    args.n_FFBs = n_FFBs
    args.n_feats = n_feats
    args.no_upsampling = no_upsampling  # True
    args.n_colors = 3
    return CSASR(args)
