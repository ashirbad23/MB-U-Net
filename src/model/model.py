import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Swish
# -------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# -------------------------
# Attention (same style)
# -------------------------
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(16, in_ch)
        self.q = nn.Conv2d(in_ch, in_ch, 1)
        self.k = nn.Conv2d(in_ch, in_ch, 1)
        self.v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)

        q = self.q(h).permute(0, 2, 3, 1).reshape(B, H*W, C)
        k = self.k(h).reshape(B, C, H*W)

        w = torch.bmm(q, k) * (C ** -0.5)
        w = torch.softmax(w, dim=-1)

        v = self.v(h).permute(0, 2, 3, 1).reshape(B, H*W, C)
        h = torch.bmm(w, v)

        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


# -------------------------
# SE (UNCHANGED behavior)
# -------------------------
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
        B, C, H, W = x.shape
        y = self.pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y, y


class IdentitySE(nn.Module):
    def forward(self, x):
        return x, None


# -------------------------
# ResBlock (with dilation)
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, dilation=1, attn=False):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            Swish(),
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.InstanceNorm2d(out_ch),
            nn.Dropout2d(dropout)
        )

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

        self.attn = AttnBlock(out_ch) if attn else nn.Identity()
        self.act = Swish()

    def forward(self, x):
        h = self.block(x)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return self.act(h)


# -------------------------
# MAIN MODEL (simple + your style)
# -------------------------
class SUnetSimple(nn.Module):
    def __init__(
        self,
        in_ch=6,
        out_ch=1,
        ch_head=64,
        num_res_blocks=2,
        dropout=0.1,
        attn=True,
        se=True
    ):
        super().__init__()

        ch_mult = (1, 2, 4, 8)

        # ---- SE (same logic, but safe) ----
        geo_ch = max(in_ch - 6, 0)
        self.se = SEBlock(geo_ch) if (se and geo_ch > 0) else IdentitySE()

        # ---- Head ----
        self.head = nn.Conv2d(in_ch, ch_head, 3, 1, 1)

        # ---- Encoder ----
        self.down = nn.ModuleList()
        chs = []
        now_ch = ch_head

        for i, mult in enumerate(ch_mult):
            out_ch_level = ch_head * mult
            for _ in range(num_res_blocks):
                self.down.append(
                    ResBlock(
                        now_ch,
                        out_ch_level,
                        dropout,
                        dilation=2**i,
                        attn=False
                    )
                )
                now_ch = out_ch_level
                chs.append(now_ch)

        # ---- Bottleneck ----
        self.bottleneck = nn.ModuleList([
            ResBlock(now_ch, now_ch, dropout, dilation=2**len(ch_mult), attn=attn),
            ResBlock(now_ch, now_ch, dropout, dilation=2**len(ch_mult), attn=attn),
        ])

        # ---- Decoder ----
        self.up = nn.ModuleList()

        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch_level = ch_head * mult
            for _ in range(num_res_blocks):
                self.up.append(
                    ResBlock(
                        now_ch + chs.pop(),
                        out_ch_level,
                        dropout,
                        dilation=2**i,
                        attn=(attn and i == len(ch_mult)-1)
                    )
                )
                now_ch = out_ch_level

        # ---- Tail ----
        self.tail = nn.Sequential(
            nn.Conv2d(ch_head, ch_head, 3, 1, 1),
            nn.InstanceNorm2d(ch_head),
            Swish(),
            nn.Conv2d(ch_head, out_ch, 1)
        )

    def forward(self, x):
        geo_weights = None

        # ---- SE logic (UNCHANGED behavior) ----
        if x.shape[1] > 6:
            sat = x[:, :6]
            geo = x[:, 6:]
            geo, geo_weights = self.se(geo)
            x = torch.cat([sat, geo], dim=1)

        # ---- Head ----
        h = self.head(x)

        # ---- Encoder ----
        skips = []
        for block in self.down:
            h = block(h)
            skips.append(h)

        # ---- Bottleneck ----
        for block in self.bottleneck:
            h = block(h)

        # ---- Decoder ----
        for block in self.up:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        # ---- Output ----
        return self.tail(h), geo_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 🔧 Adjust these based on your dataset
    in_channels = 6  # e.g. satellite + terrain channels
    out_channels = 1  # segmentation mask
    ch_head = 32
    num_res_blocks = 2
    dropout = 0.1

    model = SUnetSimple(
        ch_head=ch_head,
        in_ch=in_channels,
        out_ch=out_channels,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        attn=False,
        se=True
    ).to(device)

    # 🧪 Dummy input size (CHANGE if needed)
    H, W = 64, 64

    summary(model, (8, 6, H, W), device=device)

    for name, module in model.named_modules():
        if "AttnBlock" in str(type(module)):
            print(name)


