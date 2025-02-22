from architectures.nets_vit_cifar_pure import ViT, print_summary, Attention
import torch
import torch.nn as nn
from torchsummary import summary
from einops import rearrange, repeat
import functools


def recursive_getattr(obj, attr, *args):
    """递归获取嵌套属性"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class SubViT(nn.Module):
    def __init__(self, original_model, layer_indices_dict):
        super(SubViT, self).__init__()
        self.layer_indices_dict = layer_indices_dict
        self.vit = original_model
        self.dimension_reduction = nn.Linear(384, 76)
        # 修改所有注意力层的heads数量
        for module in self.vit.modules():
            if isinstance(module, Attention):
                module.heads = 2  # 将heads设置为2
        self.modify_network()

    def modify_network(self):
        # 遍历每一层,根据layer_indices_dict修改参数
        for name, param in self.vit.named_parameters():
            # 处理注意力层相关参数
            if 'to_qkv' in name:
                layer_num = name.split('.')[2]
                layer_name = f"Layer_{layer_num}_attention_head"
                if layer_name in self.layer_indices_dict:
                    # 获取选中的注意力头索引
                    selected_heads = [int(i) for i in self.layer_indices_dict[layer_name]]
                    heads = 12
                    dim_head = 64
                    
                    # 获取前一层MLP第二层选中的神经元
                    prev_layer_num = str(int(layer_num) - 1)
                    prev_layer_name = f"Layer_{prev_layer_num}_mlp_second"
                    if prev_layer_name in self.layer_indices_dict and int(layer_num) > 0:
                        prev_selected_neurons = [int(i) for i in self.layer_indices_dict[prev_layer_name]]
                    else:
                        prev_selected_neurons = list(range(76))
                        
                    # 修改这里：计算实际的输入维度
                    total_dim = param.size(0)
                    print(f"total_dim: {total_dim}")
                    num_qkv = total_dim // (heads * dim_head)
                    # 重塑参数以选择特定的注意力头
                    param_reshaped = param.view(3, heads, dim_head, -1)
                    # 只保留选中的注意力头和输入维度
                    selected_param = param_reshaped[:, selected_heads, :, :]
                    selected_param = selected_param[..., prev_selected_neurons]
                    # 重塑回原始形状
                    new_param = selected_param.reshape(-1, len(prev_selected_neurons))
                    
                    # 创建新的线性层并替换原来的层
                    module_path = '.'.join(name.split('.')[:-1])
                    parent_path = '.'.join(module_path.split('.')[:-1])
                    last_name = module_path.split('.')[-1]
                    parent_module = recursive_getattr(self.vit, parent_path)
                    new_layer = nn.Linear(len(prev_selected_neurons), len(selected_heads) * dim_head * 3, bias=False)
                    new_layer.weight.data = new_param
                    setattr(parent_module, last_name, new_layer)

            # 处理注意力层前的norm层
            elif '.0.norm' in name:
                layer_num = name.split('.')[2]
                prev_layer_num = str(int(layer_num) - 1)
                prev_layer_name = f"Layer_{prev_layer_num}_mlp_second"
                
                # 检查是否有前一层的mlp second层
                if prev_layer_name in self.layer_indices_dict and int(layer_num) > 0:
                    # 获取前一层mlp second层选中的神经元
                    selected_neurons = [int(i) for i in self.layer_indices_dict[prev_layer_name]]
                    # 创建新的LayerNorm层
                    module_path = '.'.join(name.split('.')[:-1])
                    parent_path = '.'.join(module_path.split('.')[:-1])
                    last_name = module_path.split('.')[-1]
                    parent_module = recursive_getattr(self.vit, parent_path)
                    new_norm = nn.LayerNorm(len(selected_neurons))
                    if len(param.data.shape) == 1:  # bias
                        new_norm.bias.data = param.data[selected_neurons]
                    else:  # weight
                        new_norm.weight.data = param.data[selected_neurons]
                    setattr(parent_module, last_name, new_norm)
                else:
                    selected_neurons = list(range(76))
                    module_path = '.'.join(name.split('.')[:-1])
                    parent_path = '.'.join(module_path.split('.')[:-1])
                    last_name = module_path.split('.')[-1]
                    parent_module = recursive_getattr(self.vit, parent_path)
                    new_norm = nn.LayerNorm(76)
                    if len(param.data.shape) == 1:  # bias
                        new_norm.bias.data = param.data[selected_neurons]
                    else:  # weight
                        new_norm.weight.data = param.data[selected_neurons]
                    setattr(parent_module, last_name, new_norm)

            # 处理注意力层的to_out层
            elif 'to_out.0' in name:
                layer_num = name.split('.')[2]
                if layer_num in [name.split('.')[2] for layer_name in self.layer_indices_dict if 'attention_head' in layer_name]:
                    layer_name = f"Layer_{layer_num}_attention_head"
                    selected_heads = [int(i) for i in self.layer_indices_dict[layer_name]]
                    if len(param.data.shape) == 2:  # weight
                        curr_layer_name = f"Layer_{layer_num}_attention_out"
                        if curr_layer_name in self.layer_indices_dict:
                            selected_outputs = [int(i) for i in self.layer_indices_dict[curr_layer_name]]
                        else:
                            selected_outputs = list(range(384))
                        # 创建新的线性层
                        module_path = '.'.join(name.split('.')[:-2])
                        parent_path = '.'.join(module_path.split('.')[:-1])
                        last_name = module_path.split('.')[-1]
                        parent_module = recursive_getattr(self.vit, parent_path)
                        new_linear = nn.Linear(len(selected_heads) * dim_head, len(selected_outputs))
                        new_linear.weight.data = param.data[selected_outputs, :][:, torch.tensor(selected_heads).repeat_interleave(dim_head)]
                        setattr(parent_module, last_name, nn.Sequential(new_linear, nn.Dropout(0.1)))

            # 处理MLP层相关参数
            elif 'fn.net.0' in name:  # MLP第一层
                layer_num = name.split('.')[2]
                layer_name = f"Layer_{layer_num}_mlp_first"
                curr_layer_name = f"Layer_{layer_num}_attention_out"
                if layer_name in self.layer_indices_dict:
                    selected_neurons = [int(i) for i in self.layer_indices_dict[layer_name]]
                    if curr_layer_name in self.layer_indices_dict:
                        prev_selected_outputs = [int(i) for i in self.layer_indices_dict[curr_layer_name]]
                    else:
                        prev_selected_outputs = list(range(384))
                    
                    # 修改这里：正确访问MLP层
                    module_path = '.'.join(name.split('.')[:4])  # transformer.layers.0.1
                    parent_module = recursive_getattr(self.vit, module_path)  # 获取PreNorm模块
                    new_linear = nn.Linear(len(prev_selected_outputs), len(selected_neurons))
                    if len(param.data.shape) == 1:  # bias
                        new_linear.bias.data = param.data[selected_neurons]
                    else:  # weight
                        new_linear.weight.data = param.data[selected_neurons, :][:, prev_selected_outputs]
                    parent_module.fn.net[0] = new_linear

            elif 'fn.net.3' in name:  # MLP第二层
                layer_num = name.split('.')[2]
                layer_name = f"Layer_{layer_num}_mlp_second"
                layer_name_first = f"Layer_{layer_num}_mlp_first"
                if layer_name in self.layer_indices_dict:
                    selected_neurons = [int(i) for i in self.layer_indices_dict[layer_name]]
                    selected_neurons_first = [int(i) for i in self.layer_indices_dict[layer_name_first]]
                    
                    # 修改这里：正确访问MLP层
                    module_path = '.'.join(name.split('.')[:4])  # transformer.layers.0.1
                    parent_module = recursive_getattr(self.vit, module_path)  # 获取PreNorm模块
                    new_linear = nn.Linear(len(selected_neurons_first), len(selected_neurons))
                    if len(param.data.shape) == 1:  # bias
                        new_linear.bias.data = param.data[selected_neurons]
                    else:  # weight
                        new_linear.weight.data = param.data[selected_neurons, :][:, selected_neurons_first]
                    parent_module.fn.net[3] = new_linear

            # 处理MLP层前的norm层
            elif '.1.norm' in name and layer_num in [name.split('.')[2] for layer_name in self.layer_indices_dict if 'mlp' in layer_name]:
                layer_num = name.split('.')[2]
                curr_layer_name = f"Layer_{layer_num}_attention_out"
                if curr_layer_name in self.layer_indices_dict:
                    selected_outputs = [int(i) for i in self.layer_indices_dict[curr_layer_name]]
                    # 创建新的LayerNorm层
                    module_path = '.'.join(name.split('.')[:-1])
                    parent_path = '.'.join(module_path.split('.')[:-1])
                    last_name = module_path.split('.')[-1]
                    parent_module = recursive_getattr(self.vit, parent_path)
                    new_norm = nn.LayerNorm(len(selected_outputs))
                    if len(param.data.shape) == 1:  # bias
                        new_norm.bias.data = param.data[selected_outputs]
                    else:  # weight
                        new_norm.weight.data = param.data[selected_outputs]
                    setattr(parent_module, last_name, new_norm)

            # 处理最后的mlp_head层
            elif 'mlp_head.0' in name:
                last_layer_num = 6
                last_layer_name = f"Layer_{last_layer_num}_mlp_second"
                if last_layer_name in self.layer_indices_dict:
                    selected_neurons = [int(i) for i in self.layer_indices_dict[last_layer_name]]
                    # 创建新的LayerNorm层
                    new_norm = nn.LayerNorm(len(selected_neurons))
                    if len(param.data.shape) == 1:  # bias
                        new_norm.bias.data = param.data[selected_neurons]
                    else:  # weight
                        new_norm.weight.data = param.data[selected_neurons]
                    self.vit.mlp_head[0] = new_norm
            
            elif 'mlp_head.1' in name:
                last_layer_num = 6
                last_layer_name = f"Layer_{last_layer_num}_mlp_second"
                if last_layer_name in self.layer_indices_dict:
                    selected_neurons = [int(i) for i in self.layer_indices_dict[last_layer_name]]
                    # 创建新的线性层
                    new_linear = nn.Linear(len(selected_neurons), self.vit.mlp_head[1].out_features)
                    if len(param.data.shape) == 1:  # bias
                        new_linear.bias.data = param.data
                    else:  # weight
                        new_linear.weight.data = param.data[:, selected_neurons]
                    self.vit.mlp_head[1] = new_linear

    def forward(self, x):
        # 1. 通过patch embedding层
        x = self.vit.to_patch_embedding(x)
        
        # 2. 获取batch size和序列长度
        b, n, _ = x.shape
        
        # 3. 准备和添加分类token
        cls_tokens = repeat(self.vit.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 4. 添加位置编码
        x += self.vit.pos_embedding[:, :(n + 1)]
        
        # 5. 应用dropout
        x = self.vit.dropout(x)

        # 新增：维度降低层
        x = self.dimension_reduction(x)
        print(f"x.shape: {x.shape}")
        # 6. 通过transformer层
        x = self.vit.transformer(x)
        
        # 7. 池化操作：使用mean pooling或者取CLS token
        x = x.mean(dim=1) if self.vit.pool == 'mean' else x[:, 0]
        
        # 8. 通过潜在层
        x = self.vit.to_latent(x)
        
        # 9. 最后通过MLP头得到分类结果
        x = self.vit.mlp_head(x)
        
        return x

def compare_parameters(model1, model2):
    """比较两个模型对应位置参数的差异"""
    diff_dict = {}
    
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"参数名不匹配: {name1} vs {name2}"
        
        # 计算参数差异
        diff = (param1 - param2).abs()
        
        # 处理注意力层
        if 'to_qkv' in name1:
            # qkv投影层的形状为 [2304, 384]
            heads = 12  # 从archs.log可以看出heads=12
            dim_head = 384 // heads  # 384/12=32
            
            # 修正参数重塑
            param_reshaped = diff.view(2304, -1)  # [2304, 384]
            head_diff = param_reshaped.mean(dim=1)  # 对每个head取平均
            
            for h in range(heads):
                diff_dict[f"{name1}_head_{h}"] = {
                    'diff': head_diff[h].item(),
                    'type': 'attention_head',
                    'layer': name1.split('.')[2] if 'transformer.layers' in name1 else 'other'
                }
        # 处理注意力层的to_out.0层 [384, 768]
        elif 'to_out.0' in name1:
            if len(param1.shape) == 2:
                neuron_diff = diff.mean(dim=1)  # 对每个输出神经元取平均
                for n in range(384):
                    diff_dict[f"{name1}_neuron_{n}"] = {
                        'diff': neuron_diff[n].item(),
                        'type': 'attention_out',  # 区分注意力输出层
                        'layer': name1.split('.')[2] if 'transformer.layers' in name1 else 'other'
                    }

        # 处理MLP层和线性层
        elif len(param1.shape) == 2:  # 只处理权重矩阵
            if 'fn.net.0' in name1:  # MLP第一层 [1536, 384]
                neuron_diff = diff.mean(dim=1)  # 对每个输出神经元取平均
                for n in range(1536):
                    diff_dict[f"{name1}_neuron_{n}"] = {
                        'diff': neuron_diff[n].item(),
                        'type': 'mlp_first',  # 区分第一层MLP
                        'layer': name1.split('.')[2] if 'transformer.layers' in name1 else 'other'
                    }
            elif 'fn.net.3' in name1:  # MLP第二层 [384, 1536]
                neuron_diff = diff.mean(dim=1)
                for n in range(384):
                    diff_dict[f"{name1}_neuron_{n}"] = {
                        'diff': neuron_diff[n].item(),
                        'type': 'mlp_second',  # 区分第二层MLP
                        'layer': name1.split('.')[2] if 'transformer.layers' in name1 else 'other'
                    }
    
    return diff_dict

if __name__ == '__main__':
    
    # 创建两个相同架构的ViT模型
    vit_model1 = ViT(
        image_size=32,
        patch_size=2,
        num_classes=10,
        dim=384,
        depth=7,
        heads=12,
        mlp_dim=384 * 4,
        dropout=0.1,
        emb_dropout=0.1
    )
    
    vit_model2 = ViT(
        image_size=32,
        patch_size=2,
        num_classes=10,
        dim=384,
        depth=7,
        heads=12,
        mlp_dim=384 * 4,
        dropout=0.1,
        emb_dropout=0.1
    )

    # 比较参数差异
    diff_dict = compare_parameters(vit_model1, vit_model2)
    
    # 按层和类型分组
    layer_groups = {}
    for name, info in diff_dict.items():
        layer = info['layer']
        comp_type = info['type']
        layer_name = f"Layer_{layer}_{comp_type}"
        
        if layer_name not in layer_groups:
            layer_groups[layer_name] = []
        layer_groups[layer_name].append((name, info))

    # 打印分析结果
    print("\n=== 每层差异分析 ===")
    for layer_name, components in sorted(layer_groups.items()):
        if layer_name == 'other':
            continue
            
        # 按差异大小排序
        sorted_components = sorted(components, key=lambda x: x[1]['diff'], reverse=True)
        
        # 计算统计信息
        diffs = [info['diff'] for _, info in sorted_components]
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        
        # 获取差异最大的前20%组件
        num_to_show = max(1, int(len(components) * 0.2))
        top_components = sorted_components[:num_to_show]
        top_indices = [name.split('_')[-1] for name, _ in top_components]
        
        # 将layer_name和top_indices存入字典
        if 'layer_indices_dict' not in locals():
            layer_indices_dict = {}
        layer_indices_dict[layer_name] = top_indices
        
        # 打印信息
        print(f"\n{layer_name}:")
        print(f"组件总数: {len(components)}")
        print(f"差异最大的组件索引: {top_indices}")

    # 创建子网络
    subnet = SubViT(vit_model1, layer_indices_dict)

    # for name, param in subnet.named_parameters():
    #     print(f"Name: {name}, Shape: {param.shape}")
    #     print("-" * 100)

    img = torch.randn(1, 3, 32, 32)  # CIFAR-10图像大小为32x32
    preds = subnet(img)  # 使用CIFAR-10的ViT模型
    print_summary(torch.randn(1, 3, 32, 32), subnet.cpu())
    summary(subnet.cpu(), input_size=(3, 32, 32), device='cpu')