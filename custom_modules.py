import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- 必须补上这一行

class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        # 初始时不建立卷积层，避开 YOLO 初始化时的通道缩放计算
        self.c1 = c1
        self.ca = None
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 动态获取当前 Tensor 的真实通道数 (c)
        c = x.shape[1]
        
        # 如果是第一次运行或者通道对不上，现场重新造一个注意力层
        if self.ca is None or getattr(self, '_last_c', None) != c:
            self.ca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, c // 16 if c > 16 else 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // 16 if c > 16 else 1, c, 1, bias=False),
                nn.Sigmoid()
            ).to(x.device)
            self._last_c = c # 记录当前的通道数

        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = self.sa(torch.cat([avg_out, max_out], dim=1))
        return x * spatial


class MobileNetBlock(nn.Module):
    """Depthwise-separable block with dynamic channel adaptation."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.dw = None
        self.pw = None
        self.bn1 = None
        self.bn2 = None
        self.act = nn.ReLU6(inplace=True)
        self._last_c = None

    def _build(self, x):
        c = x.shape[1]
        if self._last_c == c and self.dw is not None:
            return
        pad = self.kernel_size // 2
        self.dw = nn.Conv2d(c, c, self.kernel_size, 1, pad, groups=c, bias=False).to(device=x.device, dtype=x.dtype)
        self.bn1 = nn.BatchNorm2d(c).to(device=x.device, dtype=x.dtype)
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=False).to(device=x.device, dtype=x.dtype)
        self.bn2 = nn.BatchNorm2d(c).to(device=x.device, dtype=x.dtype)
        self._last_c = c

    def forward(self, x):
        self._build(x)
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class DySample(nn.Module):
    def __init__(self, c1, scale=2):
        super().__init__()
        self.scale = scale
        # 初始时不建立 offset_conv，因为 c1 会被 YOLO 缩放逻辑搞乱
        self.offset_conv = None 

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 动态初始化：第一次运行或通道数变化时重新构建卷积层
        if self.offset_conv is None or self.offset_conv.weight.shape[1] != c:
            self.offset_conv = nn.Conv2d(c, 2 * self.scale * self.scale, 1).to(device=x.device, dtype=x.dtype)
            nn.init.constant_(self.offset_conv.weight, 0)
            nn.init.constant_(self.offset_conv.bias, 0)

        # 生成偏移量
        offset = self.offset_conv(x)
        offset = F.pixel_shuffle(offset, self.scale)
        
        # 生成采样网格
        # 使用 torch.meshgrid 配合 indexing='ij'
        h_range = torch.linspace(-1, 1, h * self.scale, device=x.device, dtype=x.dtype)
        w_range = torch.linspace(-1, 1, w * self.scale, device=x.device, dtype=x.dtype)
        grid = torch.stack(torch.meshgrid([h_range, w_range], indexing='ij'), -1)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1).to(dtype=x.dtype) # 使用 expand 节省内存
        
        # 归一化偏移量并应用到网格
        # 偏移量需要根据特征图大小进行归一化，否则采样会跑偏
        offset = offset.permute(0, 2, 3, 1)
        grid = grid + offset
        
        # 进行双线性采样
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class HybridAttention(nn.Module):
    """论文式混合注意力：通道分支（SE/CBAM 通道：Avg+Max 池化 + 共享 MLP）+ 空间分支（CBAM 空间：通道维 Avg/Max 拼接 + 卷积）。

    与 Woo et al. CBAM 顺序一致：F' = Mc(F)⊙F，F'' = Ms(F')⊙F'。对应文中 Mc、Ms、逐元素乘 ⊙。
    超参默认与文中表一致：通道压缩比 r=16，空间卷积核 3×3。
    通道 MLP 按运行时真实通道数惰性构建，以兼容 YOLO 宽度缩放后 c 与 yaml 中 c1 不完全一致的情况。
    输出端采用可学习门控融合，在“原特征”和“注意力特征”之间自适应平衡。
    这比强残差更稳，通常更利于在不改训练参数时维持召回。
    """

    def __init__(self, c1, reduction=16, kernel_size=3):
        super().__init__()
        self.c1 = c1
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.avg_pool_c = nn.AdaptiveAvgPool2d(1)
        self.max_pool_c = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = None
        self._last_c = None
        pad = kernel_size // 2
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        # alpha in [0, 1]: 0 -> identity, 1 -> fully attended
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _build_channel_mlp(self, x):
        c = x.shape[1]
        if self.channel_mlp is not None and self._last_c == c:
            return
        hidden = max(c // self.reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
        ).to(device=x.device, dtype=x.dtype)
        self._last_c = c

    def forward(self, x):
        x_in = x
        self._build_channel_mlp(x)
        # Mc(F)：AvgPool 与 MaxPool 经同一 MLP 再相加后 Sigmoid（CBAM 通道注意力）
        mc = torch.sigmoid(
            self.channel_mlp(self.avg_pool_c(x)) + self.channel_mlp(self.max_pool_c(x))
        )
        x = x * mc
        # Ms(F')：在通道加权后的特征上做空间注意力（通道维 mean / max，拼接后卷积）
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        ms = torch.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        out = x * ms
        # Learnable blend between identity and attended features.
        a = torch.sigmoid(self.alpha)
        return (1.0 - a) * x_in + a * out