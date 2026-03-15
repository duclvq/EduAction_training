import torch
import torch.nn as nn


class PatchEmbed(nn.Module):

    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Dùng Convolution để cắt patch và embed cùng lúc
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Tạo Q, K, V trong 1 lần tính toán
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Tính qkv: [Batch, N, 3*C] -> tách thành 3 phần q, k, v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]


        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBackbone(nn.Module):


    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))  # +1 cho cls token (nếu dùng)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Chuỗi các khối Transformer xếp chồng lên nhau
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 1. Patch Embedding
        x = self.patch_embed(x)  # [B, N, C]

        if self.pos_embed.shape[1] == x.shape[1] + 1:
            x = x + self.pos_embed[:, 1:, :]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape lại thành dạng ảnh [Batch, C, H, W] để đưa vào Head
        B, N, C = x.shape
        H_grid, W_grid = self.patch_embed.grid_size
        x = x.transpose(1, 2).reshape(B, C, H_grid, W_grid)
        return x



# SIMPLE DECODER HEAD
class TopDownSimpleHead(nn.Module):

    def __init__(self, in_channels, out_channels, num_deconv_layers=2, num_deconv_filters=(256, 256),
                 num_deconv_kernels=(4, 4)):
        super().__init__()
        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
            in_channels
        )
        # Lớp cuối cùng dự đoán heatmap cho từng khớp
        self.final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, in_channels):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], stride=2)
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_channels = planes
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, stride=2):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x


class ViTPose(nn.Module):
    def __init__(self, model_size='base', num_joints=17, img_size=(256, 192)):
        super().__init__()

        if model_size == 'base':
            embed_dim = 768
            depth = 12
            num_heads = 12
        self.backbone = ViTBackbone(
            img_size=img_size,
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=True
        )

        self.keypoint_head = TopDownSimpleHead(
            in_channels=embed_dim,
            out_channels=num_joints,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4)
        )

    def forward(self, x):
        # 1. Feature Extraction
        features = self.backbone(x)

        # 2. Heatmap Prediction
        heatmaps = self.keypoint_head(features)
        return heatmaps
