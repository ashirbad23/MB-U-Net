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
        self.bottleneck = ResBlock(chs[-1], chs[-1] * 2, dropout, True)

        # Decoder
        self.up_blocks = nn.ModuleList()
        rev_chs = list(reversed(chs))
        prev_ch = chs[-1] * 2
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


class MultiBranchUNet(nn.Module):
    def __init__(self, ch_head, in_ch, out_ch, num_levels, dropout, bands_used=None, attn=False):
        super().__init__()

        # ---- BAND GROUPS (FIXED DEFINITIONS) ----
        self.all_sat = [0, 1, 2, 3, 4, 5]
        self.all_dem = [6, 7, 8, 9]
        self.all_curv = [10, 11, 12, 13, 14, 15, 16, 17]

        self.bands_used = bands_used if bands_used is not None else list(range(in_ch))

        # ---- FILTER ACTIVE CHANNELS ----
        self.sat_idx = [i for i in self.all_sat if i in self.bands_used]
        self.dem_idx = [i for i in self.all_dem if i in self.bands_used]
        self.curv_idx = [i for i in self.all_curv if i in self.bands_used]

        self.sat_ch = len(self.sat_idx)
        self.dem_ch = len(self.dem_idx)
        self.curv_ch = len(self.curv_idx)

        assert self.sat_ch + self.dem_ch + self.curv_ch == len(self.bands_used)

        # ---- CHANNEL SIZES ----
        # ---- PER-BRANCH HEAD CHANNELS ----
        # ch_head can be:
        #   int        -> same width for all branches
        #   [sat, dem, curv]

        if isinstance(ch_head, int):
            self.sat_head = ch_head
            self.dem_head = ch_head
            self.curv_head = ch_head
        else:
            assert len(ch_head) == 3, "ch_head must be int or [sat_head, dem_head, curv_head]"

            self.sat_head = ch_head[0]
            self.dem_head = ch_head[1]
            self.curv_head = ch_head[2]

        # ---- DECODER CHANNEL SIZES (based on spectral width) ----
        # The decoder and final fused representation use the spectral branch width
        # as the reference scale.
        base_mult = [1, 2, 4, 8, 16, 32]
        ch_mult = base_mult[:num_levels]

        chs = [self.sat_head * m for m in ch_mult]

        # ---- BUILD ENCODERS ONLY IF NEEDED ----
        if self.sat_ch > 0:
            self.enc_sat = self._make_encoder(
                self.sat_ch,
                self.sat_head,
                chs,
                dropout,
                attn
            )

        if self.dem_ch > 0:
            self.enc_dem = self._make_encoder(
                self.dem_ch,
                self.dem_head,
                chs,
                dropout,
                attn
            )

        if self.curv_ch > 0:
            self.enc_curv = self._make_encoder(
                self.curv_ch,
                self.curv_head,
                chs,
                dropout,
                attn
            )

        # ---- COUNT ACTIVE BRANCHES ----
        self.num_branches = sum([self.sat_ch > 0, self.dem_ch > 0, self.curv_ch > 0])

        # Small residual cross-modal information leak
        self.leak_alpha = nn.Parameter(
            torch.full((len(chs),), 0.1)
        )

        # ---- BOTTLENECK ----
        self.bottleneck_fuse = nn.Conv2d(chs[-1] * self.num_branches, chs[-1], 1)
        self.bottleneck = ResBlock(chs[-1], chs[-1] * 2, dropout, True)

        # ---- SKIP FUSION ----
        self.skip_fuse = nn.ModuleList([
            nn.Conv2d(ch * self.num_branches, ch, kernel_size=1) for ch in chs
        ])

        # ---- EARLY CROSS-MODAL LEAK ----
        # At each encoder level:
        #   concatenate all branch skips
        #   compress to common width
        #   inject a small residual into every branch
        self.leak_fuse = nn.ModuleList([
            nn.Conv2d(ch * self.num_branches, ch, kernel_size=1)
            for ch in chs
        ])

        # ---- DECODER ----
        self.up_blocks = nn.ModuleList()
        rev_chs = list(reversed(chs))
        prev_ch = chs[-1] * 2

        for ch in rev_chs:
            self.up_blocks.append(UpBlock(prev_ch, ch, dropout, attn))
            prev_ch = ch

        self.tail = nn.Sequential(
            nn.Conv2d(chs[0], chs[0], 3, padding=1),
            nn.InstanceNorm2d(chs[0]),
            Swish(),
            nn.Conv2d(chs[0], out_ch, 1)
        )

    def _make_encoder(self, in_ch, ch_head, chs, dropout, attn):
        layers = nn.ModuleList()
        head = nn.Conv2d(in_ch, ch_head, 3, padding=1)

        prev_ch = ch_head
        for ch in chs:
            layers.append(DownBlock(prev_ch, ch, dropout, attn))
            prev_ch = ch

        return nn.ModuleDict({
            "head": head,
            "blocks": layers
        })

    def _forward_encoder(self, x, encoder):
        x = encoder["head"](x)
        skips = []

        for down in encoder["blocks"]:
            s, x = down(x)
            skips.append(s)

        return x, skips

    def forward(self, x):
        # =====================================================
        # INITIALIZE BRANCH INPUTS
        # =====================================================

        branch_x = {}
        branch_skips = {}

        if self.sat_ch > 0:
            branch_x["sat"] = self.enc_sat["head"](x[:, self.sat_idx])
            branch_skips["sat"] = []

        if self.dem_ch > 0:
            branch_x["dem"] = self.enc_dem["head"](x[:, self.dem_idx])
            branch_skips["dem"] = []

        if self.curv_ch > 0:
            branch_x["curv"] = self.enc_curv["head"](x[:, self.curv_idx])
            branch_skips["curv"] = []

        active_names = list(branch_x.keys())

        # =====================================================
        # LEVEL-BY-LEVEL ENCODING WITH INFORMATION LEAK
        # =====================================================

        for level in range(len(self.skip_fuse)):

            # ---------------------------------------------
            # Run one down block for each active branch
            # ---------------------------------------------
            level_skips = []

            for name in active_names:
                encoder = getattr(self, f"enc_{name}")
                skip, out = encoder["blocks"][level](branch_x[name])

                branch_x[name] = out
                branch_skips[name].append(skip)

                level_skips.append(skip)

            # ---------------------------------------------
            # Cross-modal information leak
            # ---------------------------------------------
            if len(level_skips) > 1:

                shared = torch.cat(level_skips, dim=1)
                shared = self.leak_fuse[level](shared)

                for name in active_names:
                    branch_skips[name][-1] = (
                            branch_skips[name][-1]
                            + self.leak_alpha[level] * shared
                    )

        # =====================================================
        # Collect final features and skips
        # =====================================================

        features = [branch_x[name] for name in active_names]
        skips_all = [branch_skips[name] for name in active_names]

        # ---- BOTTLENECK ----
        x = torch.cat(features, dim=1)
        x = self.bottleneck_fuse(x)
        x = self.bottleneck(x)

        # ---- SKIP FUSION ----
        fused_skips = []
        depth = len(skips_all[0])

        for i in range(depth):
            to_concat = [skips[i] for skips in skips_all]
            fused = torch.cat(to_concat, dim=1)
            fused = self.skip_fuse[i](fused)
            fused_skips.append(fused)

        # ---- DECODE ----
        for up in self.up_blocks:
            skip = fused_skips.pop()
            x = up(x, skip)

        out = self.tail(x)
        return out, None


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 18
    out_channels = 1
    ch_head = [16, 8, 8]
    num_levels = 4  # 🔥 THIS controls depth now (But don't increase the depth beyond 5 for 64x64 input)
    dropout = 0.1

    model = MultiBranchUNet(
        ch_head=ch_head,
        in_ch=in_channels,
        out_ch=out_channels,
        num_levels=num_levels,
        dropout=dropout,
        attn=False
    ).to(device)

    H, W = 128, 128

    summary(model, (in_channels, H, W), batch_size=16)
    print(model.leak_alpha.data)
