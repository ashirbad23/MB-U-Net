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
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
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
        B, C, H, W = x.shape

        y = self.pool(x).view(B, C)  # [B, C]
        y = self.fc(y).view(B, C, 1, 1)  # [B, C, 1, 1]

        return x * y, y  # <-- return weights also


class IdentitySE(nn.Module):
    def forward(self, x):
        return x, None


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, attn=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.Dropout2d(dropout)
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

        self.swish = Swish()

    def forward(self, x):
        h = self.block(x)
        s = self.shortcut(x)
        h = h + s
        h = self.attn(h)
        return self.swish(h)


class SUnet(nn.Module):
    def __init__(self, ch_head, in_ch, out_ch, num_res_blocks, dropout, attn=True, se=True):
        super().__init__()
        self.head = nn.Conv2d(in_ch, ch_head, 3, 1, 1)
        self.down_blocks = nn.ModuleList()
        chs = [ch_head]

        now_ch = ch_head
        ch_mult = [1, 2, 4, 8]
        for i, mult in enumerate(ch_mult):
            im_ch = ch_head // mult
            print(im_ch)
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(now_ch, im_ch, dropout, attn))
                now_ch = im_ch
                chs.append(now_ch)
        self.bottle_neck = nn.ModuleList([
            ResBlock(now_ch, now_ch, dropout, attn),
            ResBlock(now_ch, now_ch, dropout, attn)
        ])
        chs = chs[:-1]
        mid_ch = now_ch

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            im_ch = ch_head // mult
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResBlock(chs.pop() + now_ch, im_ch, dropout, attn))
                now_ch = im_ch
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.Conv2d(ch_head * 2, ch_head * 2, 3, 1, 1),
            nn.InstanceNorm2d(ch_head * 2),
            Swish(),
            nn.Conv2d(ch_head * 2, out_ch, 1, 1, 0)
        )

        if se:
            self.se = SEBlock(12)
        else:
            self.se = IdentitySE()

    def forward(self, x):
        sat = x[:, :6, :, :]
        geo = x[:, 6:, :, :]

        geo, geo_weights = self.se(geo)
        x = torch.cat([sat, geo], dim=1)

        h = self.head(x)

        skips = [h]

        for block in self.down_blocks:
            h = block(h)
            skips.append(h)

        for block in self.bottle_neck:
            h = block(h)

        feature = h

        skips = skips[:-1]

        i = 1
        for block in self.up_blocks:
            skip = skips.pop()
            i += 1
            h = torch.cat([h, skip], dim=1)
            h = block(h)
            if i % 2 == 0:
                feature = torch.cat([feature, h], dim=1)

        out = self.tail(feature)
        return out, geo_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 🔧 Adjust these based on your dataset
    in_channels = 18  # e.g. satellite + terrain channels
    out_channels = 1  # segmentation mask
    ch_head = 128
    num_res_blocks = 2
    dropout = 0.1

    model = SUnet(
        ch_head=ch_head,
        in_ch=in_channels,
        out_ch=out_channels,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        attn=True,
        se=True
    ).to(device)

    # 🧪 Dummy input size (CHANGE if needed)
    H, W = 64, 64

    summary(model, (in_channels, H, W), batch_size=4)


