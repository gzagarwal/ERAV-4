import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, p_drop=0.0, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # stride on 3x3 conv (torchvision style)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p_drop) if p_drop and p_drop > 0 else nn.Identity()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Net(nn.Module):
    """
    ResNet-50 layout:
    stem -> conv2_x(3) -> conv3_x(4) -> conv4_x(6) -> conv5_x(3) -> GAP -> 1x1 conv head
    """

    def __init__(self, in_channels=3, num_classes=100, p_drop=0.05):
        super().__init__()
        # ----- Stem -----
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # -> 56x56

        # ===== conv2_x (64->256), 3 blocks, no spatial downsample in first block =====
        self.ds2_1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.b2_1 = Bottleneck(64, 64, stride=1, p_drop=p_drop, downsample=self.ds2_1)
        self.b2_2 = Bottleneck(256, 64, stride=1, p_drop=p_drop)
        self.b2_3 = Bottleneck(256, 64, stride=1, p_drop=p_drop)

        # ===== conv3_x (128->512), 4 blocks, first with stride=2 =====
        self.ds3_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
        )
        self.b3_1 = Bottleneck(256, 128, stride=2, p_drop=p_drop, downsample=self.ds3_1)
        self.b3_2 = Bottleneck(512, 128, stride=1, p_drop=p_drop)
        self.b3_3 = Bottleneck(512, 128, stride=1, p_drop=p_drop)
        self.b3_4 = Bottleneck(512, 128, stride=1, p_drop=p_drop)

        # ===== conv4_x (256->1024), 6 blocks, first with stride=2 =====
        self.ds4_1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024),
        )
        self.b4_1 = Bottleneck(512, 256, stride=2, p_drop=p_drop, downsample=self.ds4_1)
        self.b4_2 = Bottleneck(1024, 256, stride=1, p_drop=p_drop)
        self.b4_3 = Bottleneck(1024, 256, stride=1, p_drop=p_drop)
        self.b4_4 = Bottleneck(1024, 256, stride=1, p_drop=p_drop)
        self.b4_5 = Bottleneck(1024, 256, stride=1, p_drop=p_drop)
        self.b4_6 = Bottleneck(1024, 256, stride=1, p_drop=p_drop)

        # ===== conv5_x (512->2048), 3 blocks, first with stride=2 =====
        self.ds5_1 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(2048),
        )
        self.b5_1 = Bottleneck(
            1024, 512, stride=2, p_drop=p_drop, downsample=self.ds5_1
        )
        self.b5_2 = Bottleneck(2048, 512, stride=1, p_drop=p_drop)
        self.b5_3 = Bottleneck(2048, 512, stride=1, p_drop=p_drop)

        # ---- Head: GAP -> 1x1 logits (no flatten before Conv2d) ----
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(2048, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        # Stem
        x = self.convblock1(x)
        x = self.pool1(x)  # ensure 56x56 before conv2_x

        # conv2_x
        x = self.b2_1(x)
        x = self.b2_2(x)
        x = self.b2_3(x)

        # conv3_x
        x = self.b3_1(x)
        x = self.b3_2(x)
        x = self.b3_3(x)
        x = self.b3_4(x)

        # conv4_x
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)
        x = self.b4_6(x)

        # conv5_x
        x = self.b5_1(x)
        x = self.b5_2(x)
        x = self.b5_3(x)

        # head
        x = self.avgpool(x)  # (B, 2048, 1, 1)
        x = self.fc(x)  # (B, num_classes, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, num_classes)
        return x
