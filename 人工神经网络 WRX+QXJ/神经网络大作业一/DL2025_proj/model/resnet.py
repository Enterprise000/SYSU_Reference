'''
Reference: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
'''


from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

try:
  from torch.hub import load_state_dict_from_url
except ImportError:
  from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torchvision.models._utils import _ovewrite_named_param
import torchvision.models

__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d", "wide_resnet50_2", "wide_resnet101_2"]

# 这是一个字典，存储了不同ResNet变种的预训练模型的URL。每个键是一个ResNet的变种名称（例如 resnet18, resnet50），
# 每个值是对应的预训练模型文件的下载链接。通过这些链接，用户可以直接下载预训练的模型权重。
model_urls = {
  'resnet18':'https://download.pytorch.org/models/resnet18-f37072fd.pth',
  'resnet34':'https://download.pytorch.org/models/resnet34-b627a593.pth',
  'resnet50':'https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
  'resnet101':'https://download.pytorch.org/models/resnet101-cd907fc2.pth',
  'resnet152':'https://download.pytorch.org/models/resnet152-f82ba261.pth',
  'resnext50_32x4d':"https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
  'resnext101_32x8d':"https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
  'resnext101_64x4d':"https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
  'wide_resnet50_2':"https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
  'wide_resnet101_2':"https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
}


# 这两个是辅助函数，用于创建标准的卷积层。
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 是ResNet中的基本模块，称为“BasicBlock”，通常用于较浅的网络
# 该模块包含两个卷积层：conv1 和 conv2。每个卷积后面跟着一个批量归一化（Batch Normalization）和ReLU激活函数。
# 包含两次3x3卷积操作，并使用残差连接。
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 实现了ResNet中的瓶颈（Bottleneck）结构的神经网络模块，通常用于ResNet的较深版本
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 插入dropout(forward方法里也改了）
        self.dropout = nn.Dropout(p = 0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    # 该方法构建网络的每一层，每层由多个残差块组成
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    # 前向传播（forward 和 _forward_impl 方法）
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # 添加了dropout
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(arch, pretrained=False, progress=True, **kwargs):
    model = ResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # model.load_state_dict(state_dict,strict=False)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet model as in
    https://arxiv.org/abs/1512.03385

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: Resnet-18 network
    """

    return _resnet('resnet18',
                   pretrained, progress,
                   block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    """Construct 34 layer Resnet model as in
    https://arxiv.org/abs/1512.03385

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: Resnet-34 network
    """

    return _resnet('resnet34',
                   pretrained, progress,
                   block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   **kwargs)

# 构建一个 50 层的 ResNet 模型
def resnet50(pretrained=False, progress=True, **kwargs):
    """Construct 50 layer Resnet model as in
    https://arxiv.org/abs/1512.03385

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: Resnet-50 network
    """

    return _resnet('resnet50',
                   pretrained, progress,
                   block=Bottleneck,
                   layers=[3, 4, 6, 3],
                   # norm_layer=nn.Identity,
                   **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    """Construct 101 layer Resnet model as in
    https://arxiv.org/abs/1512.03385

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: Resnet-101 network
    """

    return _resnet('resnet101',
                   pretrained, progress,
                   block=Bottleneck,
                   layers=[3, 4, 23, 3],
                   **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    """Construct 152 layer Resnet model as in
    https://arxiv.org/abs/1512.03385

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: Resnet-152 network
    """

    return _resnet('resnet152',
                   pretrained, progress,
                   block=Bottleneck,
                   layers=[3, 8, 36, 3],
                   **kwargs)

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Construct ResNeXt-50 32x4d model as in
    https://arxiv.org/abs/1611.05431

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returnsgroups
        nn.Module: ResNeXt-50 32x4d network
    """
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet('resnext50_32x4d',
                   pretrained, progress,
                   block=Bottleneck,
                   layers=[3, 4, 6, 3],
                   **kwargs)

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Construct ResNeXt-101 32x8d model as in
    https://arxiv.org/abs/1611.05431

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returnsgroups
        nn.Module: ResNeXt-101 32x8d network
    """
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet('resnext101_32x8d',
                   pretrained, progress,
                   block=Bottleneck,
                   layers=[3, 4, 23, 3],
                   **kwargs)

def resnext101_64x4d(pretrained=False, progress=True, **kwargs):
    """Construct ResNeXt-101 64x4d model as in
    https://arxiv.org/abs/1611.05431

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returnsgroups
        nn.Module: ResNeXt-101 64x4d network
    """
    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet('resnext101_64x4d',
                   pretrained, progress,
                   block=Bottleneck,
                   layers=[3, 4, 23, 3],
                   **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    """Construct Wide ResNet-50-2 model as in
    https://arxiv.org/abs/1605.07146

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returnsgroups
        nn.Module: Wide ResNet-50-2 network
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet('wide_resnet50_2',
                   pretrained, progress,
                   block=Bottleneck,
                   layers=[3, 4, 6, 3],
                   **kwargs)

def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    """Construct Wide ResNet-101-2 model as in
    https://arxiv.org/abs/1605.07146

    Args:
        pretrained (bool): If True, returns a model pre-trained
        progress (bool): If True, displays a progress bar of the download to stderr

    Returnsgroups
        nn.Module: Wide ResNet-101-2 network
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet('wide_resnet101_2',
                   pretrained, progress,
                   block=Bottleneck,
                   layers=[3, 4, 23, 3],
                   **kwargs)


class ResNetWithFeatures(nn.Module):
    def __init__(self, base_name='resnet50', num_classes=200, pretrained=True):
        super().__init__()
        if base_name == 'resnet50':
            base_model = resnet50(pretrained = pretrained)
            feature_dim = 2048
        elif base_name == 'resnet34':
            base_model = resnet34(pretrained = pretrained)
            feature_dim = 512
        elif base_name == 'resnet18':
            base_model = resnet18(pretrained = pretrained)
            feature_dim = 512
        elif base_name == 'resnet101':
            base_model = resnet101(pretrained = pretrained)
            feature_dim = 2048
        elif base_name == 'resnet152':
            base_model = resnet152(pretrained = pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported resnet variant: {base_name}")

        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # 去掉最后fc
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        out = self.fc(features)
        if return_features:
            return out, features
        return out


# class ResNetFineTune(nn.Module):
#     def __init__(self,num_classes = 200, pretrained=True,freeze_backbone=True):
#         super().__init__()
#         self.backbone = models.resnet50(pretrained=pretrained)
#         if freeze_backbone:
#             print("only last layer")
#             for name, param in self.backbone.named_parameters():
#                 if "fc" not in name:
#                     param.requires_grad = False
#         in_features = self.backbone.fc.in_features
#         self.backbone.fc = nn.Linear(in_features, num_classes)

#     def forward(self,x):
#         return self.backbone(x)