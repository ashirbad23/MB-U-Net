import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(16, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h).permute(0, 2, 3, 1).reshape(B, H * W, C)
        k = self.proj_k(h).reshape(B, C, H * W)
        w = torch.bmm(q, k) * (C ** -0.5)
        w = F.softmax(w, dim=-1)

        v = self.proj_v(h).permute(0, 2, 3, 1).reshape(B, H * W, C)
        h = torch.bmm(w, v).reshape(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=2):
        super().__init__()
        hidden = max(in_ch // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y, y


class IdentitySE(nn.Module):
    def forward(self, x):
        return x, None


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, attn=False):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            Swish(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.Dropout2d(dropout)
        )

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()
        self.swish = Swish()

    def forward(self, x):
        h = self.block(x) + self.shortcut(x)
        h = self.attn(h)
        return self.swish(h)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, attn=False):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, dropout, attn)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        h = self.res(x)
        p = self.pool(h)
        return h, p


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, attn=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.res = ResBlock(in_ch, out_ch, dropout, attn)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class SUnet(nn.Module):
    def __init__(self, ch_head, in_ch, out_ch, num_levels, dropout, attn=True, se=True):
        super().__init__()

        self.head = nn.Conv2d(in_ch, ch_head, 3, padding=1)

        # channel multipliers (extendable)
        base_mult = [1, 2, 4, 8, 16, 32]
        ch_mult = base_mult[:num_levels]
        chs = [ch_head * m for m in ch_mult]

        # Encoder
        self.down_blocks = nn.ModuleList()
        prev_ch = ch_head
        for ch in chs:
            self.down_blocks.append(DownBlock(prev_ch, ch, dropout, attn))
            prev_ch = ch

        # Bottleneck
        self.bottleneck = ResBlock(chs[-1], chs[-1]*2, dropout, attn)

        # Decoder
        self.up_blocks = nn.ModuleList()
        rev_chs = list(reversed(chs))
        prev_ch = chs[-1]*2
        for ch in rev_chs:
            self.up_blocks.append(UpBlock(prev_ch, ch, dropout, attn))
            prev_ch = ch

        self.tail = nn.Sequential(
            nn.Conv2d(chs[0], chs[0], 3, padding=1),
            nn.InstanceNorm2d(chs[0]),
            Swish(),
            nn.Conv2d(chs[0], out_ch, 1)
        )

        self.se = SEBlock(in_ch - 6) if se else IdentitySE()

    def forward(self, x):
        geo_weights = None

        if x.shape[1] > 6:
            sat = x[:, :6]
            geo = x[:, 6:]
            geo, geo_weights = self.se(geo)
            x = torch.cat([sat, geo], dim=1)

        x = self.head(x)

        skips = []
        for down in self.down_blocks:
            s, x = down(x)
            skips.append(s)

        x = self.bottleneck(x)

        for up in self.up_blocks:
            skip = skips.pop()
            x = up(x, skip)

        out = self.tail(x)

        return out, geo_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 18
    out_channels = 1
    ch_head = 32
    num_levels = 4   # 🔥 THIS controls depth now (But don't increase the depth beyond 5 for 64x64 input)
    dropout = 0.1

    model = SUnet(
        ch_head=ch_head,
        in_ch=in_channels,
        out_ch=out_channels,
        num_levels=num_levels,
        dropout=dropout,
        attn=True,
        se=True
    ).to(device)

    H, W = 64, 64

    summary(model, (in_channels, H, W), batch_size=4)

