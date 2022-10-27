import torch.nn as nn
from .method.adl import ADL
from .method.dhl import DHL


# _INSERT_POSITION = [[], [], [], [], [2]]


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = self.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(extra_x + out)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return self.relu(x + output)



class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# layers=[3, 4, 6, 3]
class ResNet50WSOL(nn.Module):
    def __init__(self, block=Bottleneck, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNet50WSOL, self).__init__()

        self.stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64
        self.layers = [3, 4, 6, 3]

        self.wsol = kwargs['wsol']

        self.insert_position = kwargs['insert_position']
        if self.wsol == 'adl':
            self.adl_drop_rate = kwargs['adl_drop_rate']
            self.adl_threshold = kwargs['adl_drop_threshold']
        elif self.wsol == 'dhl':
            self.dhl_importance_rate = kwargs['dhl_importance_rate']
            self.dhl_drop_or_highlight_rate = kwargs['dhl_drop_or_highlight_rate']
            self.dhl_drop_threshold = kwargs['dhl_drop_threshold']
            self.dhl_highlight_threshold = kwargs['dhl_highlight_threshold']

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, self.layers[0],
                                       stride=1,
                                       split=self.insert_position[1])
        self.layer2 = self._make_layer(block, 128, self.layers[1],
                                       stride=2,
                                       split=self.insert_position[2])
        self.layer3 = self._make_layer(block, 256, self.layers[2],
                                       stride=self.stride_l3,
                                       split=self.insert_position[3])
        self.layer4 = self._make_layer(block, 512, self.layers[3],
                                       stride=1,
                                       split=self.insert_position[4])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        if return_cam:
            feature_map = x.detach().clone()
            cam_weights = self.fc.weight[labels]
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            return cams

        return logits, x

    def _make_layer(self, block, planes, blocks, stride, split=None):
        layers = self._layer(block, planes, blocks, stride)
        for pos in reversed(split):
            if self.wsol == 'adl':
                layers.insert(pos + 1, ADL(self.adl_drop_rate, self.adl_threshold))
            elif self.wsol == 'dhl':
                layers.insert(pos + 1, DHL(self.dhl_importance_rate, self.dhl_drop_or_highlight_rate,
                                           self.dhl_drop_threshold, self.dhl_highlight_threshold))
            else:
                raise TypeError("WSOL type unknown({})".format(self.wsol))
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = self.get_downsampling_layer(self.inplanes, block, planes, stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return layers

    def get_downsampling_layer(self, inplanes, block, planes, stride):
        outplanes = planes * block.expansion
        if stride == 1 and inplanes == outplanes:
            return
        else:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )


def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


# layers=[2, 2, 2, 2]
class ResNet18Adl(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super(ResNet18Adl, self).__init__()
        self.adl_drop_rate = kwargs['adl_drop_rate']
        self.adl_threshold = kwargs['adl_drop_threshold']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(BasicBlock(64, 64, 1),
                                    BasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(DownBlock(64, 128, [2, 1]),
                                    BasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(DownBlock(128, 256, [2, 1]),
                                    BasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(DownBlock(256, 512, [2, 1]),
                                    BasicBlock(512, 512, 1),
                                    ADL(adl_drop_rate=self.adl_drop_rate, adl_drop_threshold= self.adl_threshold))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feature = self.layer4(out)
        out = self.avgpool(feature)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out, feature
