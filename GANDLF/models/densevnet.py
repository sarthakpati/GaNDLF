""" Adapted from https://github.com/Vadbeg/brain_tumor_segmentation/blob/master/brain_tumor_segmentation/modules/model/dense_vnet.py"""

import numpy as np
import torch
from .modelBase import ModelBase


class DenseVNet(ModelBase):
    def __init__(self):
        super().__init__()
        in_channels = self.n_channels
        out_channels = self.n_classes

        # defaults obtained from https://niftynet.readthedocs.io/en/dev/_modules/niftynet/network/dense_vnet.html#DenseVNet
        kernel_size = [5, 3, 3]
        num_downsample_channels = [24, 24, 24]
        num_skip_channels = [12, 24, 24]
        units = [5, 10, 10]
        growth_rate = [4, 8, 16]

        self.dfs_blocks = torch.nn.ModuleList()
        for i in range(3):
            self.dfs_blocks.append(
                DownsampleWithDfs(
                    in_channels=in_channels,
                    downsample_channels=num_downsample_channels[i],
                    skip_channels=num_skip_channels[i],
                    kernel_size=kernel_size[i],
                    units=units[i],
                    growth_rate=growth_rate[i],
                    conv=self.Conv,
                    batch_norm_layer=self.BatchNorm,
                    constant_pad_layer=self.ConstantPad,
                )
            )
            in_channels = num_downsample_channels[i] + units[i] * growth_rate[i]

        self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.upsample_2 = torch.nn.Upsample(scale_factor=4, mode="trilinear")

        self.out_conv = ConvBlock(
            in_channels=sum(num_skip_channels),
            out_channels=out_channels,
            kernel_size=3,
            batch_norm=True,
            preactivation=True,
            conv=self.Conv,
            batch_norm_layer=self.BatchNorm,
            constant_pad_layer=self.ConstantPad,
        )
        self.upsample_out = torch.nn.Upsample(scale_factor=2, mode="trilinear")

    def forward(self, x):
        x, skip_1 = self.dfs_blocks[0](x)
        x, skip_2 = self.dfs_blocks[1](x)
        _, skip_3 = self.dfs_blocks[2](x)

        skip_2 = self.upsample_1(skip_2)
        skip_3 = self.upsample_2(skip_3)

        out = self.out_conv(torch.cat([skip_1, skip_2, skip_3], 1))
        out = self.upsample_out(out)

        return out


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        batch_norm=True,
        preactivation=False,
        conv=torch.nn.Conv3d,
        batch_norm_layer=torch.nn.BatchNorm3d,
        constant_pad_layer=torch.nn.ConstantPad3d,
    ):
        super().__init__()

        if dilation != 1:
            raise NotImplementedError()

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = constant_pad_layer(tuple([padding % 2, padding - padding % 2] * 3), 0)
        else:
            pad = constant_pad_layer(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(),
                pad,
                conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers = [batch_norm_layer(in_channels)] + layers
        else:
            layers = [
                pad,
                conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers.append(batch_norm_layer(out_channels))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DenseFeatureStack(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        units,
        growth_rate,
        kernel_size,
        dilation=1,
        batch_norm=True,
        batchwise_spatial_dropout=False,
        conv=torch.nn.Conv3d,
        batch_norm_layer=torch.nn.BatchNorm3d,
        constant_pad_layer=torch.nn.ConstantPad3d,
    ):
        super().__init__()

        self.units = torch.nn.ModuleList()
        for _ in range(units):
            if batchwise_spatial_dropout:
                raise NotImplementedError

            self.units.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=growth_rate,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=1,
                    batch_norm=batch_norm,
                    preactivation=True,
                    conv=conv,
                    batch_norm_layer=batch_norm_layer,
                    constant_pad_layer=constant_pad_layer,
                )
            )
            in_channels += growth_rate

    def forward(self, x):
        feature_stack = [x]

        for unit in self.units:
            inputs = torch.cat(feature_stack, 1)
            out = unit(inputs)
            feature_stack.append(out)

        return torch.cat(feature_stack, 1)


class DownsampleWithDfs(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        downsample_channels,
        skip_channels,
        kernel_size,
        units,
        growth_rate,
        conv,
        batch_norm_layer,
        constant_pad_layer,
    ):
        super().__init__()

        self.downsample = ConvBlock(
            in_channels=in_channels,
            out_channels=downsample_channels,
            kernel_size=kernel_size,
            stride=2,
            batch_norm=True,
            preactivation=True,
        )
        self.dfs = DenseFeatureStack(
            downsample_channels,
            units,
            growth_rate,
            3,
            batch_norm=True,
            conv=conv,
            batch_norm_layer=batch_norm_layer,
            constant_pad_layer=constant_pad_layer,
        )
        self.skip = ConvBlock(
            in_channels=downsample_channels + units * growth_rate,
            out_channels=skip_channels,
            kernel_size=3,
            batch_norm=True,
            preactivation=True,
            conv=conv,
            batch_norm_layer=batch_norm_layer,
            constant_pad_layer=constant_pad_layer,
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.dfs(x)
        x_skip = self.skip(x)

        return x, x_skip
