import torch
import torch.nn as nn
import torch.nn.functional as F


# --- cheap 3x3 depthwise + 1x1 pointwise ---
class DWSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, p_drop=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False
            ),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.net(x)


class Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, p_drop=0.05):
        super().__init__()

        # ---- Block 1: 32x32 -> 16x16, 64ch ----
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Conv2d(
                16, 32, 3, stride=2, padding=1, bias=False
            ),  # downsample to 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Conv2d(64, 64, 1, bias=False),  # cheap bottleneck
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )

        # (optional) light transition (kept from your code; parameter-cheap)
        self.transition1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        # ---- Block 2: stay 16x16 initially, then to 8x8, reach 128ch ----
        self.conv_64_96 = nn.Sequential(  # 64 -> 96 (3x3)
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )

        # Replacements for your heavy 3x3s
        self.c31 = DWSeparableConv(96, 128, stride=1, p_drop=p_drop)  # 96 -> 128
        self.c35 = DWSeparableConv(128, 128, stride=1, p_drop=p_drop)  # 128 -> 128
        self.c38 = DWSeparableConv(128, 128, stride=2, p_drop=p_drop)  # 16x16 -> 8x8

        # light 1x1 to mix channels (was Conv2d-41 in your summary)
        self.mix_1x1 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # ---- Head: GAP -> 1x1 logits ----
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Conv2d(128, num_classes, 1, bias=False)

    def forward(self, x, debug=False):
        x = self.convblock1(x)  # -> (B, 64, 16, 16)
        if debug:
            print("after block1:", x.shape)

        x = self.transition1(x)  # -> (B, 64, 16, 16)
        if debug:
            print("after trans1:", x.shape)

        x = self.conv_64_96(x)  # -> (B, 96, 16, 16)
        if debug:
            print("after 64->96:", x.shape)

        x = self.c31(x)  # -> (B, 128, 16, 16)
        x = self.c35(x)  # -> (B, 128, 16, 16)
        x = self.c38(x)  # -> (B, 128, 8, 8)
        if debug:
            print("after DW stack:", x.shape)

        x = self.mix_1x1(x)  # -> (B, 128, 8, 8)
        x = self.gap(x)  # -> (B, 128, 1, 1)
        x = self.head(x)  # -> (B, 10, 1, 1)
        x = x.view(x.size(0), -1)  # logits (B, 10)
        return x
