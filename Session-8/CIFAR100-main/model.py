import torch
import torch.nn as nn

class ResNet34(nn.Module):
    """ResNet34 architecture adapted for CIFAR-100.
    
    Key modifications from standard ResNet34:
    - Smaller initial conv layer (3x3 instead of 7x7)
    - No initial maxpool
    - Added dropout before final FC layer
    - Uses label smoothing in training
    """
    def __init__(self, num_classes=100):
        super(ResNet34, self).__init__()
        self.in_planes = 64

        # First layer adapted for CIFAR-100's 32x32 images
        # Using 3x3 conv instead of 7x7, stride=1 instead of 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet-34 block configuration: [3, 4, 6, 3]
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2) 
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        """Creates one stage of residual blocks.
        
        Args:
            planes: Number of output channels
            num_blocks: Number of residual blocks in this stage
            stride: Stride for first block (for downsampling)
        """
        layers = []

        # First block handles stride and channel changes
        layers.append(self._residual_block(self.in_planes, planes, stride))
        self.in_planes = planes

        # Remaining blocks (stride = 1)
        for _ in range(1, num_blocks):
            layers.append(self._residual_block(planes, planes, stride=1))

        return nn.Sequential(*layers)

    def _residual_block(self, in_planes, planes, stride):
        """Inline residual block (BasicBlock)."""
        shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # main conv path
        conv_path = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

        # store as tuple (conv, shortcut)
        return nn.ModuleDict({'conv': conv_path, 'shortcut': shortcut})

    def _forward_layer(self, x, layer):
        for block in layer:
            out = block['conv'](x)
            out += block['shortcut'](x)
            x = self.relu(out)
        return x

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self._forward_layer(x, self.layer1)
        x = self._forward_layer(x, self.layer2)
        x = self._forward_layer(x, self.layer3)
        x = self._forward_layer(x, self.layer4)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # More memory efficient than view
        x = self.dropout(x)  # Add dropout before final layer
        x = self.fc(x)
        return x
