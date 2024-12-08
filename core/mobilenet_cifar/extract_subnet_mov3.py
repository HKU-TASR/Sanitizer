import torch
from contextlib import redirect_stdout
from torch import nn
from torchsummary import summary

from architectures.nets_MobileNetV3 import MobileNetV3_Small
from thop import profile, clever_format
import functools


def recursive_setattr(obj, attr, value):
    """递归设置嵌套属性"""
    pre, _, post = attr.rpartition('.')
    return setattr(recursive_getattr(obj, pre) if pre else obj, post, value)

def recursive_getattr(obj, attr, *args):
    """递归获取嵌套属性"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


# 创建一个新的子网络类 （含抽取后门子网络）
class SubMoblieNetV3(nn.Module):
    def __init__(self, original_model, selected_kernels, selected_kernels_linear):
        super(SubMoblieNetV3, self).__init__()
        self.selected_kernels = selected_kernels
        self.selected_kernels_linear = selected_kernels_linear
        self.mov3 = original_model
        self.modify_network(original_model)

    def modify_network(self, original_model):
        ################################BLOCK#####################################
        previous_out_channels = None
        previous_indices = [0, 1, 2]

        for name, indices in self.selected_kernels.items():
            name_prefix = '.'.join(name.split('.')[:-1])
            if 'skip' in name_prefix:
                continue
            module_name, layer_name = name_prefix.rsplit('.', 1) if '.' in name_prefix else ('', name_prefix)
            if module_name:
                module = recursive_getattr(self.mov3, module_name)
            else:
                module = self.mov3

            layer = getattr(module, layer_name)
            if layer_name == '1':
                bn_layer_name = layer_name.replace('1', '2')
                bn_layer = getattr(module, bn_layer_name)
            elif layer_name == '4':
                bn_layer_name = None
                bn_layer = None
            else:
                bn_layer_name = layer_name.replace('conv', 'bn')
                bn_layer = getattr(module, bn_layer_name)

            if isinstance(layer, nn.Conv2d):
                in_channels = layer.in_channels if previous_out_channels is None else previous_out_channels
                out_channels = len(indices)
                kernel_size = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                bias = layer.bias is not None

                new_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
                if layer.weight.shape[1] == 1:
                    new_layer = nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
                    previous_indices = [0]
                with torch.no_grad():
                    new_layer.weight = nn.Parameter(layer.weight[indices][:, previous_indices, :, :])
                    if bias:
                        new_layer.bias = nn.Parameter(layer.bias[indices])

                setattr(module, layer_name, new_layer)

                if isinstance(bn_layer, nn.BatchNorm2d):
                    new_bn_layer = nn.BatchNorm2d(out_channels)
                    with torch.no_grad():
                        new_bn_layer.weight = nn.Parameter(bn_layer.weight[indices])
                        new_bn_layer.bias = nn.Parameter(bn_layer.bias[indices])
                        new_bn_layer.running_mean = bn_layer.running_mean[indices]
                        new_bn_layer.running_var = bn_layer.running_var[indices]

                    setattr(module, bn_layer_name, new_bn_layer)
                    # 如果属性不存在会创建一个新的对象属性，并对属性赋值：
                previous_out_channels = out_channels
                previous_indices = indices
        ################################BLOCK#####################################

        #################################SHORTCUT####################################
        previous_out_channels = None
        previous_indices = [0]

        for name, indices in self.selected_kernels.items():
            name_prefix = '.'.join(name.split('.')[:-1])
            if 'skip' not in name_prefix:
                continue
            module_name, layer_name = name_prefix.rsplit('.', 1) if '.' in name_prefix else ('', name_prefix)
            if module_name:
                module = recursive_getattr(self.mov3, module_name)
            else:
                module = self.mov3

            layer = getattr(module, layer_name)
            if layer_name == '0':
                bn_layer_name = layer_name.replace('0', '1')
                bn_layer = getattr(module, bn_layer_name)
            elif layer_name == '2':
                bn_layer_name = layer_name.replace('2', '3')
                bn_layer = getattr(module, bn_layer_name)


            if isinstance(layer, nn.Conv2d):
                in_channels = len(selected_kernels['bneck.0.conv3.weight']) if previous_out_channels is None else previous_out_channels
                out_channels = len(indices)
                kernel_size = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                bias = layer.bias is not None

                new_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
                if layer.weight.shape[1] == 1:
                    new_layer = nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
                    previous_indices = [0]
                with torch.no_grad():
                    new_layer.weight = nn.Parameter(layer.weight[indices][:, previous_indices, :, :])
                    if bias:
                        new_layer.bias = nn.Parameter(layer.bias[indices])

                setattr(module, layer_name, new_layer)

                if isinstance(bn_layer, nn.BatchNorm2d):
                    new_bn_layer = nn.BatchNorm2d(out_channels)
                    with torch.no_grad():
                        new_bn_layer.weight = nn.Parameter(bn_layer.weight[indices])
                        new_bn_layer.bias = nn.Parameter(bn_layer.bias[indices])
                        new_bn_layer.running_mean = bn_layer.running_mean[indices]
                        new_bn_layer.running_var = bn_layer.running_var[indices]

                    setattr(module, bn_layer_name, new_bn_layer)
                    # 如果属性不存在会创建一个新的对象属性，并对属性赋值：
                previous_out_channels = out_channels
                previous_indices = indices


        ################################SHORTCUT#####################################

        ################################FC LAYER#####################################

        output = nn.Linear(len(selected_kernels['conv2.weight']), len(selected_kernels_linear['linear3.weight']))
        output1 = nn.Linear(len(selected_kernels_linear['linear3.weight']), len(original_model.linear4.bias))
        new_bn_layer = nn.BatchNorm1d(len(selected_kernels_linear['linear3.weight']))
        with torch.no_grad():
            output.weight = nn.Parameter(original_model.linear3.weight[selected_kernels_linear['linear3.weight'], :][:, selected_kernels['conv2.weight']])

            new_bn_layer.weight = nn.Parameter(original_model.bn3.weight[selected_kernels_linear['linear3.weight']])
            new_bn_layer.bias = nn.Parameter(original_model.bn3.bias[selected_kernels_linear['linear3.weight']])
            new_bn_layer.running_mean = original_model.bn3.running_mean[selected_kernels_linear['linear3.weight']]
            new_bn_layer.running_var = original_model.bn3.running_var[selected_kernels_linear['linear3.weight']]

            output1.weight = nn.Parameter(original_model.linear4.weight[:, selected_kernels_linear['linear3.weight']])
            output1.bias = nn.Parameter(original_model.linear4.bias)

        setattr(self.mov3, 'linear3', output)
        setattr(self.mov3, 'linear4', output1)
        setattr(self.mov3, 'bn3', new_bn_layer)

    def forward(self, x):
        return self.mov3(x)


def print_summary(input_shape, model):
    # 计算 MACs 和参数量
    macs, params = profile(model, inputs=(input_shape,))

    # 计算 FLOPs
    flops = 2 * macs

    # 格式化输出
    macs, params, flops = clever_format([macs, params, flops], "%.3f")

    print(f"MACs/MAdds: {macs}")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")


if __name__ == '__main__':

    net1 = MobileNetV3_Small(num_classes=10)
    net2 = MobileNetV3_Small(num_classes=10)
    # summary(net1.cpu(), input_size=(3, 32, 32), device='cpu')

    # print_summary(torch.randn(1, 3, 32, 32), net1.cpu())

    # modual_temp = recursive_getattr(net1, 'bneck.1.skip')
    # layer_temp = getattr(modual_temp, '0')
    # print(layer_temp.weight.data)

    # 获取两个模型的参数字典
    original_params = {name: param.detach().clone() for name, param in net1.named_parameters()}
    finetuned_params = {name: param.detach().clone() for name, param in net2.named_parameters()}

    # 计算每个卷积核的权重变化
    param_diff = {}
    for name in original_params:
        if ('conv' in name or 'se' in name) and len(original_params[name].shape) == 4:
            param_diff[name] = (original_params[name] - finetuned_params[name]).abs().mean(dim=[1, 2, 3])
            # 在输入通道、高度和宽度维度上求平均值，得到每个卷积核的平均变化。这一步的结果是一个形状为 [out_channels] 的张量，
            # 其中每个值表示对应输出通道的卷积核的平均变化量。
        if 'skip' in name and len(original_params[name].shape) == 4:
            param_diff[name] = (original_params[name] - finetuned_params[name]).abs().mean(dim=[1, 2, 3])

    # 计算线性层的权重变化
    param_diff_linear = {}
    for name in original_params:
        if 'linear' in name and 'weight' in name:
            param_diff_linear[name] = (original_params[name] - finetuned_params[name]).abs().sum(dim=1)



    # 打印每个卷积层中每个卷积核的权重变化
    # for name, diff in param_diff.items():
    #     print(f"Layer: {name}")
    #     for i, value in enumerate(diff):
    #         print(f"  Kernel {i}: {value.item()}")

    # 找出变化最大的卷积核
    # num_kernels_to_select = max(1, int(len(diff) * args.topK_ratio1))  # 可以根据需要调整
    selected_kernels = {}
    for name, diff in param_diff.items():
        num_kernels_to_select = max(1, int(len(diff) * 0.2))
        _, indices = torch.topk(diff, num_kernels_to_select)  # 对 indices 进行排序，目前来看跟顺序没有关系；
        sorted_indices = torch.sort(indices).values
        selected_kernels[name] = sorted_indices.tolist()

    # 找出变化最大的神经元
    # num_kernels_to_select = max(1, int(len(diff) * args.topK_ratio1))  # 可以根据需要调整
    selected_kernels_linear = {}
    for name, diff in param_diff_linear.items():
        num_kernels_to_select = max(1, int(len(diff) * 0.2))
        _, indices = torch.topk(diff, num_kernels_to_select)  # 对 indices 进行排序，目前来看跟顺序没有关系；
        sorted_indices = torch.sort(indices).values
        selected_kernels_linear[name] = sorted_indices.tolist()

    # 打印展示每层变化最大的卷积核的索引，这里是kernels，不是neurons
    # for name, indices in selected_kernels.items():
    #     print(f"Layer: {name}, Top {len(indices)} Changed Kernel Indices: {indices}")

    print('*' * 200)

    # 打印展示每层变化最大的卷积核的索引，这里是kernels，不是neurons
    # for name, indices in selected_kernels_linear.items():
    #     print(f"Layer: {name}, Top {len(indices)} Changed Neuron Indices: {indices}")

    subnet = SubMoblieNetV3(net1, selected_kernels, selected_kernels_linear)
    # for name, param in subnet.named_parameters():
    #     print(f"Name: {name}, Shape: {param.shape}")
    #     print("-" * 100)

    summary(subnet.cpu(), input_size=(3, 32, 32), device='cpu')
    print_summary(torch.randn(1, 3, 32, 32), subnet.cpu())
    # modual_temp = recursive_getattr(subnet, 'mov3.bneck.1.skip')
    # layer_temp = getattr(modual_temp, '0')
    # print(layer_temp.weight.data)
    # print(subnet)
    # subnet.eval()
    # x = torch.randn(3, 3, 32, 32)
    # y = subnet(x)
    # print(y.size())

    # 打印展示每层变化最大的卷积核的索引，这里是kernels，不是neurons
    # for name, indices in selected_kernels.items():
    #     print(f"Layer: {name}, Top {len(indices)} Changed Kernel Indices: {indices}")

    # for name, indices in selected_kernels.items():
    #     print('1:' + name)
    #     name_prefix = '.'.join(name.split('.')[:-1])
    #     print('2:' + name_prefix)
    #     if 'shortcut.0' in name_prefix:
    #         continue
    #     module_name, layer_name = name_prefix.rsplit('.', 1) if '.' in name_prefix else ('', name_prefix)
    #     print('3:' + module_name)
    #     print('4:' + layer_name)
    #     if module_name:
    #         module = recursive_getattr(net1, module_name)
    #     else:
    #         module = net1
    #
    #     layer = getattr(module, layer_name)
    #     print('5:' + str(layer.weight.shape[1]))
