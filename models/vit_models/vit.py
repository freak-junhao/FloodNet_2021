import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # x = self.transformer(x)
        y = self.transformer(x)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = y.mean(dim=1) if self.pool == 'mean' else y[:, 0]
        x = self.to_latent(x)

        x = self.mlp_head(x)

        y = y[:, 1:, :]
        return x, y


class Out_Line(nn.Module):
    def __init__(self, in_channel, dim, class_num):
        super().__init__()

        self.conv = nn.Conv1d(in_channel, 1, 1)
        self.liner = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, class_num)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.liner(x)
        return x


class Count_Vit(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = ViT(
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            num_classes=config['classes'],
            dim=config['dim'],
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=config['mlp_dim'],
            dropout=config['dropout'],
            emb_dropout=config['emb_dropout']
        )

        num_patches = (config['image_size'] // config['patch_size']) ** 2
        self.output = Out_Line(num_patches, config['dim'], config['num_classes'])

    def forward(self, data):
        x, y = self.backbone(data)
        x = torch.sigmoid(x)
        y = self.output(y)
        return x, y


if __name__ == '__main__':
    config = {
        'image_size': 256,
        'patch_size': 32,
        'classes': 2,
        'num_classes': 51,
        'dim': 1024,
        # 'dim': 2048,
        'depth': 6,
        'heads': 16,
        'mlp_dim': 2048,
        'dropout': 0.1,
        'emb_dropout': 0.1
    }
    # model = ViT(image_size=256,
    #             patch_size=32,
    #             num_classes=1000,
    #             dim=1024,
    #             depth=6,
    #             heads=16,
    #             mlp_dim=2048,
    #             dropout=0.1,
    #             emb_dropout=0.1)
    model = Count_Vit(config)
    print(model)

    model_input = torch.randn(1, 3, 256, 256)
    out = model(model_input)
    print(out.shape)