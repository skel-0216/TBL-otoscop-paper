import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4  # channel expansion ratio of the bottleneck block

    def __init__(self, in_channels, out_channels, stride=1, downsample=False, dropout_rate=0.3):
        super().__init__()
        # 1x1 Conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3 Conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 Conv (expansion)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropout:
            out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)
        return out


class ResNet(nn.Module):
    def __init__(self, config, output_dim, dropout_rate=0.3, zero_init_residual=False):
        """
        config: (block, n_blocks, channels)
        output_dim: number of output classes
        dropout_rate: dropout probability inside the blocks
        """
        super().__init__()
        block, n_blocks, channels = config
        self.in_channels = channels[0]
        
        # Set conv1 stride to 1 so a 256-px input keeps its size
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # maxpool kept (a 256-px input downsamples to ~128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Pass dropout_rate into the blocks of each layer
        self.layer1 = self._make_layer(block, n_blocks[0], channels[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], channels[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], channels[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, n_blocks[3], channels[3], stride=2, dropout_rate=dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.in_channels, output_dim)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, n_blocks, channels, stride, dropout_rate):
        layers = []
        # The first block decides whether to downsample
        downsample = (self.in_channels != block.expansion * channels) or (stride != 1)
        layers.append(block(self.in_channels, channels, stride, downsample, dropout_rate))
        self.in_channels = block.expansion * channels
        for _ in range(1, n_blocks):
            layers.append(block(self.in_channels, channels, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)      # 256x256 -> 256x256 (stride=1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # 256 -> ~128
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        h = torch.flatten(x, 1)
        h = self.dropout(h)
        x = self.fc(h)
        return x, h
