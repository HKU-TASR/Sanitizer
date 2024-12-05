import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from torchsummary import summary
from thop import profile, clever_format


# This function multiplies input x by the result of ReLU6(x + 3) and divides the result by 6.
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


# This function directly uses the result of ReLU6(x + 3) and divides it by 6.
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


# The reduction factor is used for dimension reduction, default is 4. It serves as a compression factor to reduce computational complexity.
class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)  # Ensures enough feature transformation capacity even when channels are few.
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Each channel is compressed into a single value, which is the average of the features in that channel.
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),  # 1x1 convolution
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),  # 1x1 convolution
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
        # Scaling: During forward propagation, first calculate the recalibration weights for each channel via self.se(x), then multiply these weights by the original input x.


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        # Expansion: Increases the dimensionality of the channels for further processing.
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        # Depthwise convolution: groups = expand_size, making it depthwise separable. Each channel is processed separately. When groups=1, it becomes standard convolution.
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()  # If se is True, add the SE module; otherwise, use nn.Identity() to keep the output unchanged.

        # Pointwise convolution: Again using 1x1 convolution to compress expanded channels from expand_size back to out_size.
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        # Define skip path if necessary to handle residual connections.
        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                # Using bias=True here. When converting spatial dimensions and channel numbers, the bias helps in adjusting the output feature's mean more accurately.
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000, act=nn.Hardswish):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, True, 2),
            Block(3, 16, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
            Block(5, 24, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 96, act, True, 1),
            Block(5, 96, 576, 96, act, True, 1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(576, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution, batch normalization, and activation
        out = self.hs1(self.bn1(self.conv1(x)))
        # Bottleneck sequence
        out = self.bneck(out)
        # Second stage convolution, batch normalization, and activation
        out = self.hs2(self.bn2(self.conv2(out)))
        # Global average pooling and flatten
        out = self.gap(out).flatten(1)
        # Fully connected layers, batch normalization, activation, and Dropout
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))
        # Output fully connected layer
        return self.linear4(out)


def print_summary(input_shape, model):
    # Calculate MACs and parameter count
    macs, params = profile(model, inputs=(input_shape,))

    # Calculate FLOPs
    flops = 2 * macs

    # Format the output
    macs, params, flops = clever_format([macs, params, flops], "%.3f")

    print(f"MACs/MAdds: {macs}")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")


if __name__ == '__main__':
    net1 = MobileNetV3_Small(num_classes=200)
    net2 = MobileNetV3_Small(num_classes=10)

    net1.eval()
    x = torch.randn(1, 3, 32, 32)
    y = net1(x)
    print(y.size())
    summary(net2.cpu(), input_size=(3, 32, 32), device='cpu')
    print_summary(torch.randn(1, 3, 32, 32), net2.cpu())
    for name, param in net1.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}")
        print("-" * 100)

    # Print module names and layer names
    for name, module in net1.named_modules():
        print(f"Module Name: {name}")
        for param_name, param in module.named_parameters(recurse=False):
            print(f"  Layer Name: {param_name}, Shape: {param.shape}")
        print("-" * 100)

    # Print the name, value, and shape of each parameter
    for name, param in net1.named_parameters():
        if 'skip' not in name:
            print(f"Name: {name}, Shape: {param.shape}")
            print("-" * 100)
            print(f"Parameter: {param}")
            print("*" * 200)
        break
    print("*" * 200)

    for name, param in net2.named_parameters():
        print(f"Name: {name}, Shape: {len(param.shape)}")
        print("-" * 100)
        print(f"Parameter: {param}")
        print("*" * 200)
        break
