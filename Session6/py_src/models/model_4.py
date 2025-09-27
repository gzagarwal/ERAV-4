import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input shape: 1 x 28 x 28 (C x H x W)

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=0, bias=False),  # 28→26
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 26→13 (spatial size 13x13)

        # Conv Block 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(16, 18, 3, padding=0, bias=False),  # 13→11
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout(dropout_value)
        )

        # Conv Block 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(18, 20, 3, padding=0, bias=False),  # 11→9
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(dropout_value)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 9→4 (spatial size 4x4)

        # Conv Block 3
        self.convblock4 = nn.Sequential(
            nn.Conv2d(20, 10, 3, padding=0, bias=False),  # 4→2
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        )

        # Output block
        self.gap = nn.AvgPool2d(2)  # 2→1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool2(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
