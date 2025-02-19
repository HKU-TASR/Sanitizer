from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
from torch import nn
from torchsummary import summary
from thop import profile, clever_format

# ======================2.pair函数=============================#
# 辅助函数，生成元组
def pair(t):
    # 这个函数用于将输入转换为二元组
    # 如果输入t已经是tuple类型,则直接返回
    # 否则将输入t复制两次组成tuple返回
    # 例如: pair(7) 返回 (7,7), pair((7,8)) 返回 (7,8)
    return t if isinstance(t, tuple) else (t, t)


# ======================3.PreNorm （Norm Block）=============================#
# 规范化层的类封装
class PreNorm(nn.Module):
    '''
    :param  dim 输入和输出维度
            fn  前馈网络层，选择Multi-Head Attn (Attention) 和MLP二者之一
    '''

    def __init__(self, dim, fn):
        super().__init__()
        # LayerNorm: ( a - mean(last 2 dim) ) / sqrt( var(last 2 dim) )
        # 数据归一化的输入维度设定，以及保存前馈层
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    # 前向传播就是将数据归一化BN后传递给前馈层
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# ======================4.FeedForward （MLP Block）=============================#
# FeedForward层共有2个全连接层，整个结构是：FeedForward层由线性层，配合激活函数GELU和Dropout实现
# dim：输入和输出维度
'''
首先过一个全连接层
经过GELU()激活函数进行处理 -> GELU()是高斯分布的累积分布函数
nn.Dropout()，以一定概率丢失掉一些神经元，防止过拟合
然后再过一个全连接层
nn.Dropout()
'''
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ======================5.Attention （Attention Block）=============================#
# dim：输入和输出维度 encoder规定的输入的维度 也就是比如[1, 197, 768]中的768=16*16*3
# dim_head：每个头的维度
# Attention层的参数：   
'''
在自注意力机制（Self-Attention）或多头自注意力机制（Multi-Head Self-Attention）中，
主要的可训练参数包括用于计算查询（query）、键（key）和值（value）的权重矩阵，
以及在多头注意力机制中的输出投影矩阵。这些参数的数量取决于输入和输出的维度、注意力头的数量等.
'''

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()

        inner_dim = heads * dim_head  # 所有注意力头的总维度。
        project_out = not (heads == 1 and dim_head == dim)
        # 这通常用于定义投影矩阵的形状，以便将输入张量投影到适合多头注意力机制的形状。
        # 如果只有一个注意力头且每个注意力头的维度等于输入维度，那么 project_out 将为 False。否则，project_out 为 True。

        self.heads = heads
        self.scale = dim_head ** -0.5
        # 表示1/(sqrt(dim_head))用于消除误差，保证方差为1，避免向量内积过大导致的softmax将许多输出置0的情况
        # 可以看原文《attention is all you need》中关于Scale Dot-Product Attention如何抑制内积过大

        self.attend = nn.Softmax(dim=-1)
        # dim =  > 0 时，表示mask第d维度，对相同的第d维度，进行softmax
        # dim =  < 0 时，表示mask倒数第d维度，对相同的倒数第d维度，进行softmax
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # 定义一个线性层，将输入 x 投影到 3 * inner_dim 的空间
        # 生成qkv矩阵，三个矩阵被放在一起，后续会被分开

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        # 做一次线性变换在计算完所有头的注意力后，各头的输出会被拼接起来，然后通过一个线性变换（使用权重矩阵W_o来合并信息，得到最终的输出。)
        # 如果是多头注意力机制则需要进行全连接和防止过拟合，否则输出不做更改 Identity它是一个直接输出输入的恒等层。

    def forward(self, x):  # 这里一共才6行代码：

        # x 是输入的张量，通常形状为[batch_size, num_tokens, dim]，其中 batch_size 是批次大小，num_tokens 是 token 数量，dim 是输入的维度。比如[1, 197, 768]
        # 分割成q、k、v三个矩阵
        # qkv为 inner_dim * 3，其中inner_dim = heads * dim_head
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 调用 .chunk(3, dim=-1) 会将形状为 [batch_size, num_tokens, 3 * dim] 的张量沿着最后一个维度分割成 3 个形状为
        # [batch_size, num_tokens, dim] 的子张量。
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # 比如：
        # to_qkv(x)后是(1, 65, 1024*3)
        # qkv是个长度为3的元组，每个元组的大小为(1, 65, 1024)
        # rearrange后维度变成(n, heads, dim, dim_head) 即 q: [1, 16, 197, 64]  k: [1, 16, 197, 64]  v: [1, 16, 197, 64]
        # 'b n (h d) -> b h n d' 重新按思路分离出16个头（h for head），一共16组q,k,v矩阵
        # map把rearrange规则应用到qkv中的每个元素上，q、k、v维度变成(1, heads, num_tokens, dim_head)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # query * key 得到对value的注意力预测，并通过向量内积缩放防止softmax无效化部分参数
        # 因为矩阵乘法操作在 PyTorch 中是针对最后两个维度进行的，而其他维度会被广播处理。
        # 对于每个 batch 和每个 head，都会独立进行矩阵乘法，而不需要额外的处理。
        # dots的形状：[batch_size, heads, num_tokens, num_tokens]

        attn = self.attend(dots)
        # 应用 softmax：将缩放后的注意力得分通过 softmax 转换为注意力权重，使得每行的权重和为1。
        # 表示每个查询 token 对每个键 token 的得分。
        # 为什么在最后一个维度上应用？？
        # 答：对于每个查询 token，所有键 token 的注意力权重，使得这些权重和为 1，从而得到一个概率分布

        out = torch.matmul(attn, v)  # 将attn和value做点乘
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 重组张量，将heads维度重新还原，将注意力输出重排回原始维度，并通过线性层投影到最终输出。

        return self.to_out(out)


# ======================7.构建Transformer=============================#
# Transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        # 设定depth个encoder相连，并添加残差结构
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # 每次取出包含Norm-attention和Norm-mlp这两个的ModuleList，实现残差结构
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ======================8.构建ViT=============================#
# ViT
class ViT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        # image_size就是每一张图像的长和宽，通过pair函数便捷明了的表现
        # patch_size就是图像的每一个patch的长和宽
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        # 保证图像可以整除为若干个patch
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算出每一张图片会被切割为多少个patch,假设其中patch尺寸为32*32
        # 假设输入维度(64, 3, 224, 224), num_patches = 7*7 = 49
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # 每一个patch数组大小, patch_dim = 3*32*32=3072 ，经典图的举例：16×16×3 = 768
        patch_dim = channels * patch_height * patch_width

        # cls就是分类的Token， mean就是均值池化
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # embeding操作：假设输入维度(64, 3, 224, 224)，那么经过Rearange层后变成了(64, 7*7=49, 32*32*3=3072)
        self.to_patch_embedding = nn.Sequential(
            # 将图片分割为b*h*w个三通道patch，b表示输入图像数量,()表示相乘
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),

            # 下面代码经过线性全连接后，维度由(64, 49, 3072)变成(64, 49, dim)，比如dim = 1024
            nn.Linear(patch_dim, dim),
        )

        # 每张图像需要num_patches个向量进行编码
        # 位置编码(1, 50, dim) 本应该为49，但因为cls表示类别需要增加一个
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # CLS类别token,(1, 1, 128)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 设置dropout
        self.dropout = nn.Dropout(emb_dropout)

        # 初始化Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout) # encoder
        # pool默认是cls进行分类
        self.pool = pool
        self.to_latent = nn.Identity()

        # 多层感知用于将最终特征映射为num_classes个类别
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # 第一步，原始图像embedding，进行了图像切割以及线性变换，变成x->(64, 49, dim)，比如dim = 1024
        # 比如：图片的每一个通道切分为Token +  将3个channel的所有Token拉直，拉到一个1维，长度为768的向量 + 接一个线性层映射到encoder需要的维度768->1024
        x = self.to_patch_embedding(img)

        # 得到原始图像数目和单图像的patches数量, b=64, n=49
        b, n, _ = x.shape

        # (1, 1, 1024) -> (64, 1, 1024) 为每一张图像设置一个cls的token，分类头
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # 将cls token加入到数据中 -> (64, 50, 1024)
        # 将cls_token拼接到patch token中去
        x = torch.cat((cls_tokens, x), dim=1)

        # x(64, 50, 128)添加位置编码(1, 50, 128)  
        # 加位置嵌入（直接加，位置编码直接与数据相加）这一步并不会增加维度。
        # 这是一种可学习的加法，模型会自动调整合适的位置编码值。
        # : 表示选择所有批次（通常只有一个批次维度）。
        # :(n+1) 表示选择从第一个位置到第 n 个位置（包含第 n 个位置）的嵌入。
        x += self.pos_embedding[:, :(n + 1)]
        # 经过dropout层防止过拟合
        x = self.dropout(x)

        x = self.transformer(x)  # (b, 65, dim)
        # 进行均值池化
        # 0 表示选取tokens序列中的第一个位置。
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]   # (b, dim)

        x = self.to_latent(x)
        # 最终进行分类映射
        return self.mlp_head(x)


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

def ViT_cifar10():
    """
    返回一个配置好的用于CIFAR-10的ViT模型
    """
    return ViT(
        image_size=32,      # CIFAR-10图像大小为32x32
        patch_size=2,       # 使用2x2的patch size
        num_classes=10,     # CIFAR-10有10个类别
        dim=384,            # transformer编码器维度
        depth=7,            # transformer编码器层数
        heads=12,           # 注意力头数
        mlp_dim=384 * 4,    # MLP隐藏层维度,通常是dim的4倍
        dropout=0.1,        # dropout比例
        emb_dropout=0.1     # embedding dropout比例
    )

if __name__ == '__main__':

    # 创建一个用于CIFAR-10的ViT模型配置
    vit_cifar10 = ViT(
        image_size=32,      # CIFAR-10图像大小为32x32
        patch_size=2,       # 使用2x2的patch size
        num_classes=10,     # CIFAR-10有10个类别
        dim=384,            # transformer编码器维度
        depth=7,            # transformer编码器层数
        heads=12,           # 注意力头数
        mlp_dim=384 * 4,    # MLP隐藏层维度,通常是dim的4倍
        dropout=0.1,        # dropout比例
        emb_dropout=0.1     # embedding dropout比例
    )
    print("\n=== CIFAR-10 ViT model ===")
    # print(vit_cifar10)

    for name, param in vit_cifar10.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}")
        print("-" * 100)
        # print(f"Parameter: {param}")
        # print("*" * 100)

    img = torch.randn(1, 3, 32, 32)  # CIFAR-10图像大小为32x32
    preds = vit_cifar10(img)  # 使用CIFAR-10的ViT模型
    print_summary(torch.randn(1, 3, 32, 32), vit_cifar10.cpu())
    summary(vit_cifar10.cpu(), input_size=(3, 32, 32), device='cpu')
