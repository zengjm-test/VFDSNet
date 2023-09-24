from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np
from math import log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import math


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)  # , inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True,
                 relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                                   stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                                   stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels * 2, out_channels * mul, False, is_3d, bn, relu, kernel_size=3,
                                   stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_group(in_channels, out_channels, groups, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def convbn_3d_group(in_channels, out_channels, groups, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))


def convgn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.GroupNorm(4, out_channels))


def convgn_group(in_channels, out_channels, groups, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.GroupNorm(4, out_channels))


def convgn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.GroupNorm(4, out_channels))


def convgn_3d_group(in_channels, out_channels, groups, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.GroupNorm(4, out_channels))


def convbn_3d_1kk(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride),
                  padding=(0, pad, pad), bias=False),
        nn.BatchNorm3d(out_channels))


def convbn_3d_new(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                   padding=(pad, 0, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, kernel_size, 1), stride=(1, stride, 1),
                                   padding=(0, pad, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, kernel_size), stride=(1, 1, stride),
                                   padding=(0, 0, pad), bias=False),
                         nn.BatchNorm3d(out_channels))


def conv_3d_new(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                   padding=(pad, 0, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, kernel_size, 1), stride=(1, stride, 1),
                                   padding=(0, pad, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, kernel_size), stride=(1, 1, stride),
                                   padding=(0, 0, pad), bias=False))


def convTrans_3d_new(in_channels, out_channels, kernel_size, pad, output_pad, stride):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                           padding=(pad, 0, 0), output_padding=(output_pad, 0, 0), bias=False),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(1, kernel_size, 1), stride=(1, stride, 1),
                           padding=(0, pad, 0), output_padding=(0, output_pad, 0), bias=False),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(1, 1, kernel_size), stride=(1, 1, stride),
                           padding=(0, 0, pad), output_padding=(0, 0, output_pad), bias=False))


def convbn_3d_dw(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False, groups=in_channels),
                         nn.Conv3d(in_channels, out_channels, kernel_size=1),
                         nn.BatchNorm3d(out_channels))


def conv_3d_dw(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False, groups=in_channels),
                         nn.Conv3d(in_channels, out_channels, kernel_size=1))


def convTrans_3d_dw(in_channels, out_channels, kernel_size, pad, output_pad, stride):
    return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1),
                         nn.ConvTranspose3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=pad, output_padding=output_pad, bias=False, groups=out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def build_gwc_volume_cos(refimg_fea, targetimg_fea, maxdisp, num_groups):
    refimg_fea = refimg_fea / (torch.sum(refimg_fea ** 2, dim=1, keepdim=True).pow(1 / 2) + 1e-05)
    targetimg_fea = targetimg_fea / (torch.sum(targetimg_fea ** 2, dim=1, keepdim=True).pow(1 / 2) + 1e-05)
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def groupwise_correlation_norm(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = ((fea1 / (torch.norm(fea1, 2, 2, True) + 1e-05)) * (fea2 / (torch.norm(fea2, 2, 2, True) + 1e-05))).mean(
        dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume



def get_sparse(cost_volume):
    B, C, D, H, W = cost_volume.shape
    step = W / 10
    for i in range(10):
        test_cat = F.relu(get_threshold(cost_volume[:, :, :, :, int(i * step):int((i + 1) * step)]) *
                          cost_volume[:, :, :, :, int(i * step):int((i + 1) * step)] + cost_volume[:, :, :, :,
                                                                                       int(i * step):int(
                                                                                           (i + 1) * step)])
        if i == 0:
            new_cost_volume = test_cat
        else:
            new_cost_volume = torch.cat((new_cost_volume, test_cat), dim=4)
        # print(new_cost_volume.shape)
    return new_cost_volume


# 第二种方法的实现

# 1.第一分离共享操作
class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (0, (kernel_size - 1) // 2, (kernel_size - 1) // 2)  # 设置该大小的padding,能使得进行卷积后，输出的特征图的尺寸大小不变
        weight_tensor = torch.zeros(1, 1, 1, kernel_size, kernel_size)  # 定义一个1个种类,一个通道，大小为kernel_size的卷积核
        weight_tensor[0, 0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1  # 将卷积核中间那个数值设为1
        self.weight = nn.Parameter(weight_tensor)  # 将其卷积核变为可学习的参数
        # print('self.weight_shape:', self.weight.shape)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)  # 获取输入特征图的通道数
        expand_weight = self.weight.expand(inc, 1, 1, self.kernel_size,
                                           self.kernel_size).contiguous()  # 为那个共享的卷积核进行复制，复制到与输入特征图具有一样的通道数，输出也是该通道数
        # print(expand_weight.is_cuda, x.is_cuda)
        return F.conv3d(input=x, weight=expand_weight, bias=None, stride=1, padding=self.padding, dilation=1,
                        groups=inc)  # 对输入的特征图进行指定的卷积核进行卷积操作，group参数可以使得卷积核与输入的特征图的通道数不一致


# 2.平滑空洞卷积，带有残差结构
class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1, padding=(0, 1, 1)):
        super(SmoothDilatedResidualBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation * 2 - 1)  # 在空洞卷积前执行SS,分离共享操作。
        self.conv1 = nn.Conv3d(channel_num, channel_num, (1, 3, 3), 1, padding=padding, dilation=dilation,
                               groups=group,
                               bias=False)  # 空洞卷积，
        self.norm1 = nn.InstanceNorm3d(channel_num, affine=True, )  # 实例化归一层

        self.pre_conv2 = ShareSepConv(dilation * 2 - 1)
        self.conv2 = nn.Conv3d(channel_num, channel_num, (1, 3, 3), 1, padding=padding, dilation=dilation,
                               groups=group,
                               bias=False)
        self.norm2 = nn.InstanceNorm3d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
        # print(self.pre_conv1(x).shape, self.conv1(self.pre_conv1(x)).shape,
        #       self.norm1(self.conv1(self.pre_conv1(x))).shape)
        # print(y.shape)
        y = self.norm2(self.conv2(self.pre_conv2(y)))
        # print(x.shape, y.shape)
        return F.relu(x + y)


def get_new_cost_volume(cost_volume, k):
    B, C, D, H, W = cost_volume.shape
    new_volume_test = cost_volume.new_zeros([B, C, D, H, W])
    m = torch.nn.ConstantPad2d(1, 0)
    cost_volume_test = m(cost_volume)
    for b in range(0, B):
        for c in range(0, C):
            for d in range(0, D):
                for h in range(1, H):
                    for w in range(1, W):
                        clearance = cost_volume_test[b, c, d, h - k:h + k + 1,
                                    w - k:w + k + 1] - cost_volume_test[b, c, d, h, w]
                        clearance = torch.abs(clearance)
                        clearance_probability = clearance / torch.sum(clearance)
                        threshold_data = clearance_probability * torch.log2(clearance_probability)
                        threshold = torch.sum(threshold_data)
                        threshold_weight = torch.gt(threshold, torch.log2(clearance_probability)).float()
                        value = torch.sum(
                            cost_volume_test[b, c, d, h - k:h + k + 1, w - k:w + k + 1] * threshold_weight)
                        new_volume_test[b, c, d, h - 1, w - 1] = value

    return new_volume_test


def build_gwc_volume_norm(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation_norm(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                                num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation_norm(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def patch_aggregation(gwc_volume, patch_weight):
    gwc_volume_pad = torch.nn.functional.pad(gwc_volume, pad=(1, 1, 1, 1), mode="constant", value=0)
    gwc_volume_pad_unfold = gwc_volume_pad.unfold(3, 3, 1).unfold(4, 3, 1)  # [N,C,D,H,W,3,3]
    gwc_volume_pad_unfold = gwc_volume_pad_unfold.contiguous().view(gwc_volumed.shape[0], gwc_volume.shape[1],
                                                                    gwc_volume.shape[2], gwc_volume.shape[3],
                                                                    gwc_volume.shape[4], -1).permute(0, 1, 5, 2, 3, 4)
    gwc_volume_pad_unfold = patch_weight.view(gwc_volume.shape[0], gwc_volume.shape[1], 1, gwc_volume.shape[2],
                                              gwc_volume.shape[3], gwc_volume.shape[4]) * gwc_volume_pad_unfold
    gwc_volume = torch.sum(gwc_volume_pad_unfold, dim=2)
    return gwc_volume


class Build_gwc_volume_unfold(nn.Module):
    def __init__(self, maxdisp):
        self.maxdisp = maxdisp
        super(Build_gwc_volume_unfold, self).__init__()
        self.unfold = nn.Unfold((1, maxdisp), 1, 0, 1)
        self.left_pad = nn.ZeroPad2d((maxdisp - 1, 0, 0, 0))

    def forward(self, refimg_fea, targetimg_fea, num_groups):
        B, C, H, W = refimg_fea.shape
        unfolded_targetimg_fea = self.unfold(self.left_pad(targetimg_fea)).reshape(
            B, num_groups, C // num_groups, self.maxdisp, H, W)
        refimg_fea = refimg_fea.view(B, num_groups, C // num_groups, 1, H, W)
        volume = (refimg_fea * unfolded_targetimg_fea).sum(2)
        volume = torch.flip(volume, [2])
        return volume


def build_gwc_volume_v1(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, (2 * i):] = groupwise_correlation(refimg_fea[:, :, :, (2 * i):],
                                                                 targetimg_fea[:, :, :, :-(2 * i)],
                                                                 num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_ones([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class BasicBlock_gn(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock_gn, self).__init__()

        self.conv1 = nn.Sequential(convgn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convgn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class BasicBlock_groups(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, stride, downsample, pad, dilation):
        super(BasicBlock_groups, self).__init__()

        self.conv1 = nn.Sequential(convbn_group(inplanes, planes, groups, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_group(planes, planes, groups, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class attention_block(nn.Module):
    def __init__(self, channels_3d, num_heads=8, block=4):
        """
        ws 1 for stand attention
        """
        super(attention_block, self).__init__()
        self.block = block
        self.dim_3d = channels_3d
        self.num_heads = num_heads
        head_dim_3d = self.dim_3d // num_heads
        self.scale_3d = head_dim_3d ** -0.5
        self.qkv_3d = nn.Linear(self.dim_3d, self.dim_3d * 3, bias=True)
        self.final1x1 = torch.nn.Conv3d(self.dim_3d, self.dim_3d, 1)

    def forward(self, x):

        B, C, D, H0, W0 = x.shape
        pad_l = pad_t = 0
        pad_r = (self.block[2] - W0 % self.block[2]) % self.block[2]
        pad_b = (self.block[1] - H0 % self.block[1]) % self.block[1]
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        B, C, D, H, W = x.shape
        d, h, w = D // self.block[0], H // self.block[1], W // self.block[2]

        x = x.view(B, C, d, self.block[0], h, self.block[1], w, self.block[2]).permute(0, 2, 4, 6, 3, 5, 7, 1)

        qkv_3d = self.qkv_3d(x).reshape(B, d * h * w, self.block[0] * self.block[1] * self.block[2], 3, self.num_heads,
                                        C // self.num_heads).permute(3, 0, 1, 4, 2,
                                                                     5)  # [3,B,d*h*w,num_heads,blocks,C//num_heads]
        q_3d, k_3d, v_3d = qkv_3d[0], qkv_3d[1], qkv_3d[2]
        attn = (q_3d @ k_3d.transpose(-2, -1)) * self.scale_3d
        if pad_r > 0 or pad_b > 0:
            mask = torch.zeros((1, H, W), device=x.device)
            mask[:, -pad_b:, :].fill_(1)
            mask[:, :, -pad_r:].fill_(1)
            mask = mask.reshape(1, h, self.block[1], w, self.block[2]).transpose(2, 3).reshape(1, h * w, self.block[1] *
                                                                                               self.block[2])
            attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)  # 1, _h*_w, self.block*self.block, self.block*self.block
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1000.0)).masked_fill(attn_mask == 0, float(0.0))
            attn = attn + attn_mask.repeat(1, d, self.block[0], self.block[0]).unsqueeze(2)

        attn = torch.softmax(attn, dim=-1)

        x = (attn @ v_3d).view(B, d, h, w, self.num_heads, self.block[0], self.block[1], self.block[2], -1).permute(0,
                                                                                                                    4,
                                                                                                                    8,
                                                                                                                    1,
                                                                                                                    5,
                                                                                                                    2,
                                                                                                                    6,
                                                                                                                    3,
                                                                                                                    7)
        x = x.reshape(B, C, D, H, W)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :, :H0, :W0]
        return self.final1x1(x)


def disparity_variance(x, maxdisp, disparity):
    # the shape of disparity should be B,1,H,W, return is the variance of the cost volume [B,1,H,W]
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    disp_values = (disp_values - disparity) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)


def disparity_variance_confidence(x, disparity_samples, disparity):
    # the shape of disparity should be B,1,H,W, return is the uncertainty estimation
    assert len(x.shape) == 4
    disp_values = (disparity - disparity_samples) ** 2
    return torch.sum(x * disp_values, 1, keepdim=True)


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, left_input, right_input, disparity_samples):
        """
        Disparity Sample Cost Evaluator
        Description:
                Given the left image features, right iamge features and the disparity samples, generates:
                    - Warped right image features

        Args:
            :left_input: Left Image Features
            :right_input: Right Image Features
            :disparity_samples:  Disparity Samples

        Returns:
            :warped_right_feature_map: right iamge features warped according to input disparity.
            :left_feature_map: expanded left image features.
        """

        device = left_input.get_device()
        left_y_coordinate = torch.arange(0.0, left_input.size()[3], device=device).repeat(left_input.size()[2])
        # left_y_coordinate = torch.arange(0.0, left_input.size()[3]).repeat(left_input.size()[2])
        left_y_coordinate = left_y_coordinate.view(left_input.size()[2], left_input.size()[3])
        # left_y_coordinate = torch.clamp(left_y_coordinate, min=0, max=left_input.size()[3] - 1)
        left_y_coordinate = left_y_coordinate.expand(left_input.size()[0], -1, -1)

        right_feature_map = right_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])
        left_feature_map = left_input.expand(disparity_samples.size()[1], -1, -1, -1, -1).permute([1, 2, 0, 3, 4])

        disparity_samples = disparity_samples.float()

        right_y_coordinate = left_y_coordinate.expand(
            disparity_samples.size()[1], -1, -1, -1).permute([1, 0, 2, 3]) - disparity_samples

        right_y_coordinate_1 = right_y_coordinate
        right_y_coordinate = torch.clamp(right_y_coordinate, min=0, max=(right_input.size()[3] - 1))

        warped_right_feature_map = torch.gather(right_feature_map, dim=4,
                                                index=right_y_coordinate.expand(right_input.size()[1], -1, -1, -1,
                                                                                -1).permute([1, 0, 2, 3, 4]).long())

        right_y_coordinate_1 = right_y_coordinate_1.unsqueeze(1)
        warped_right_feature_map = (1 - ((right_y_coordinate_1 < 0) + \
                                         (right_y_coordinate_1 > right_input.size()[3] - 1)).float()) * \
                                   (warped_right_feature_map) + torch.zeros_like(warped_right_feature_map)

        return warped_right_feature_map, left_feature_map


def SpatialTransformer_grid(x, y, disp_range_samples):
    bs, channels, height, width = y.size()
    ndisp = disp_range_samples.size()[1]

    mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                             torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)

    # mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype),
    #                              torch.arange(0, width, dtype=x.dtype)])  # (H *W)
    mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
    mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw - disp_range_samples
    # print('##############3333333', mw, cur_disp_coords_x)

    # print("cur_disp", cur_disp, cur_disp.shape if not isinstance(cur_disp, float) else 0)
    # print("cur_disp_coords_x", cur_disp_coords_x, cur_disp_coords_x.shape)

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4)  # (B, D, H, W, 2)

    y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                             padding_mode='zeros', align_corners=True).view(bs, channels, ndisp, height,
                                                                            width)  # (B, C, D, H, W)

    # a littel difference, no zeros filling
    x_warped = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1)  # (B, C, D, H, W)
    # x_warped = x_warped.transpose(0, 1) #(C, B, D, H, W)
    #     #x1 = x2 + d >= d
    # x_warped[:, mw < disp_range_samples] = 0
    # x_warped = x_warped.transpose(0, 1) #(B, C, D, H, W)

    return y_warped, x_warped


def groupwise_correlation_4D(fea1, fea2, num_groups):
    B, C, D, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, D, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, D, H, W)
    return cost
