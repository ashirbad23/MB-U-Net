import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


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


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            Swish(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.Dropout2d(dropout)
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.swish = Swish()

    def forward(self, x):
        return self.swish(self.block(x) + self.shortcut(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        h = self.res(x)
        p = self.pool(h)
        return h, p


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.res = ResBlock(in_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class SUnet(nn.Module):
    def __init__(self, ch_head, in_ch, out_ch, dropout, se=True):
        super().__init__()

        self.head = nn.Conv2d(in_ch, ch_head, 3, 1, 1)

        # Encoder
        self.down1 = DownBlock(ch_head, 64, dropout)
        self.down2 = DownBlock(64, 128, dropout)
        self.down3 = DownBlock(128, 256, dropout)
        self.down4 = DownBlock(256, 512, dropout)

        # Bottleneck
        self.bottleneck = ResBlock(512, 1024, dropout)

        # Decoder
        self.up1 = UpBlock(1024, 512, dropout)
        self.up2 = UpBlock(512, 256, dropout)
        self.up3 = UpBlock(256, 128, dropout)
        self.up4 = UpBlock(128, 64, dropout)

        self.tail = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            Swish(),
            nn.Conv2d(64, out_ch, 1)
        )

        if se:
            self.se = SEBlock(in_ch - 6)
        else:
            self.se = IdentitySE()

    def forward(self, x):
        geo_weights = None

        if x.shape[1] > 6:
            sat = x[:, :6, :, :]
            geo = x[:, 6:, :, :]

            geo, geo_weights = self.se(geo)
            x = torch.cat([sat, geo], dim=1)

        x = self.head(x)

        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)

        x = self.bottleneck(x)

        x = self.up1(x, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)

        out = self.tail(x)

        return out, geo_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 18
    out_channels = 1
    ch_head = 64
    dropout = 0.1

    model = SUnet(
        ch_head=ch_head,
        in_ch=in_channels,
        out_ch=out_channels,
        dropout=dropout,
        se=True
    ).to(device)

    H, W = 64, 64

    summary(model, (in_channels, H, W), batch_size=4)
