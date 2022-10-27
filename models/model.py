import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from mish_activation import *
# from efficientnet_pytorch import EfficientNet

# https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065
# generalized mean pooling
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


# model = se_resnet50(num_classes=1000, pretrained='imagenet')
# model.avg_pool = GeM()

class ResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False, n_class=11):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        # 冻结前面的网络层
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # 这里考虑加入global avg_pool 和 global max_pool丰富特征
        # self.features = nn.Sequential(*list(self.model.children())[:-2])
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.reduce_layer = nn.Conv2d(4096, 2048, 1)
        # self.dropout = nn.Dropout(0.1)
        # self.fc = nn.Linear(n_features, n_class)

        # 微调后面两层
        # self.model.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        # 目前最高的是下面这种fine-tune方式,替换全连接加dropout
        self.model.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(n_features, n_class, bias=True)
        )
        # self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        # x = self.model(x)
        # batch_size = x.size()[0]
        # x = self.features(x)
        # x = self.avg_pool(x).view(batch_size, -1)
        # x = torch.cat([x1, x2], dim=1)
        # x = self.reduce_layer(x).view(batch_size, -1) # 恢复到全连接的size[batch_size, 2048]
        # x = self.fc(self.dropout(x))
        x = self.model(x)

        return x


class SeResNext(nn.Module):
    def __init__(self, model_name='seresnext50_32x4d', pretrained=False, n_class=11):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        # teacher and student method fine-tune
        self.feature = nn.Sequential(*list(self.model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, n_class)
        # 尝试此种fine-tune方式
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(0.1), # 0.1待测试
        #     nn.Linear(n_features, n_class, bias=True)
        # )

    def forward(self, x):
        # x = self.model(x)
        # teacher and student method
        batch_size = x.size(0)
        features = self.feature(x)
        pooled_features = self.avg_pool(features).view(batch_size, -1)
        x = self.fc(pooled_features)
        return features, pooled_features, x
        # return x


class SeResNet(nn.Module):
    def __init__(self, model_name='seresnext50_32x4d', pretrained=False, n_class=11):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        # teacher and student method fine-tune
        self.feature = nn.Sequential(*list(self.model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(n_features, n_class)
        # 尝试此种fine-tune方式
        self.fc = nn.Sequential(
            nn.Dropout(0.1),  # 0.1待测试
            nn.Linear(n_features, n_class, bias=True)
        )

    def forward(self, x):
        # x = self.model(x)
        # teacher and student method
        batch_size = x.size(0)
        features = self.feature(x)
        pooled_features = self.avg_pool(features).view(batch_size, -1)
        x = self.fc(pooled_features)
        return features, pooled_features, x


class ResNet(nn.Module):
    def __init__(self, model_name='resnet200d', pretrained=False, n_class=19):
        super().__init__()
        model = timm.create_model(model_name, pretrained=pretrained)
        n_features = model.fc.in_features
        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #         self.fc = nn.Sequential(
        #             # nn.Linear(n_features, n_features),
        #             nn.Dropout(0.1), # 0.1待测试
        #             nn.Linear(n_features, n_class, bias=True)
        #         )
        self.fc = nn.Sequential(
            nn.Linear(n_features, 512), Mish(), nn.BatchNorm1d(512), nn.Dropout(0.5), nn.Linear(512, n_class)
        )
        # self.fc = nn.Linear(n_features, n_class, bias=True)

    def forward(self, x):
        # x = self.model(x)
        # teacher and student method
        batch_size = x.size(0)
        features = self.feature(x)
        pooled_features = self.avg_pool(features).view(batch_size, -1)
        x = self.fc(pooled_features)

        return x, features.view(features.shape[0], -1)
        # return x


class ResNetWithFusionMask(nn.Module):
    def __init__(self, model_name='resnet200d', pretrained=False, n_class=19):
        super().__init__()
        # Naive CNN
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features

        self.stage1 = nn.Sequential(*list(self.model.children())[0:4])
        #         print(self.stage1)

        stage2 = list(self.model.children())[4]
        stage2 = list(stage2.children())
        self.stage2 = nn.Sequential(*stage2)

        stage3 = list(self.model.children())[5]
        stage3 = list(stage3.children())
        self.stage3 = nn.Sequential(*stage3)

        stage4 = list(self.model.children())[6]
        stage4 = list(stage4.children())
        self.stage4 = nn.Sequential(*stage4)

        stage5 = list(self.model.children())[7]
        stage5 = list(stage5.children())
        self.stage5 = nn.Sequential(*stage5)

        # Mask CNN
        self.model_mask = timm.create_model(model_name, pretrained=pretrained)

        self.stage1_mask = nn.Sequential(*list(self.model_mask.children())[0:4])
        #         print(self.stage1_mask)
        self.stage1_mask[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #         print(self.stage1_mask)

        stage2_mask = list(self.model_mask.children())[4]
        stage2_mask = list(stage2_mask.children())
        self.stage2_mask = nn.Sequential(*stage2_mask)

        stage3_mask = list(self.model_mask.children())[5]
        stage3_mask = list(stage3_mask.children())
        self.stage3_mask = nn.Sequential(*stage3_mask)

        stage4_mask = list(self.model_mask.children())[6]
        stage4_mask = list(stage4_mask.children())
        self.stage4_mask = nn.Sequential(*stage4_mask)

        stage5_mask = list(self.model_mask.children())[7]
        stage5_mask = list(stage5_mask.children())
        self.stage5_mask = nn.Sequential(*stage5_mask)

        self.conv1x1_sigmoid_x = nn.Sequential(
            nn.Conv2d(512, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),
            nn.Sigmoid()
        )

        self.conv1x1_sigmoid = nn.Sequential(
            nn.Conv2d(512, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),
            nn.Sigmoid()
        )

        # self.fc = nn.Linear(n_features, n_class, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # nn.Linear(n_features, n_features),
            nn.Dropout(0.1),  # 0.1待测试
            nn.Linear(n_features, n_class, bias=True)
        )
        self.fc2 = nn.Sequential(
            # nn.Linear(n_features, n_features),
            nn.Dropout(0.1),  # 0.1待测试
            nn.Linear(n_features, n_class, bias=True)
        )

    def forward(self, x, x_mask):
        # x = self.model(x)
        # teacher and student method
        lamda = 0.4
        batch_size = x.size(0)
        # stage1
        x = self.stage1(x)
        x_mask = self.stage1_mask(x_mask)
        tmp1, tmp2 = x, x_mask
        x = (1 - lamda) * tmp1 + lamda * tmp2
        x_mask = (1 - lamda) * tmp2 + lamda * tmp1

        # stage2
        x = self.stage2(x)
        x_mask = self.stage2_mask(x_mask)
        x = x + x_mask
        # stage3
        x = self.stage3(x)
        x_mask = self.stage3_mask(x_mask)
        x = x + x_mask
        # stage4
        x = self.stage4(x)
        x_mask = self.stage4_mask(x_mask)
        x = x + x_mask
        # stage5
        x = self.stage5(x)
        x_mask = self.stage5_mask(x_mask)
        # x_mask conv1x1+sigmoid
        tmp1 = self.conv1x1_sigmoid_x(x)
        tmp2 = self.conv1x1_sigmoid(x_mask)

        features1 = x + x * tmp2
        features2 = x_mask + x_mask * tmp1

        pooled_features1 = self.avg_pool(features1).view(batch_size, -1)
        pooled_features2 = self.avg_pool(features2).view(batch_size, -1)
        x = self.fc(pooled_features1)
        x_mask = self.fc2(pooled_features2)

        return x, x_mask, pooled_features1, pooled_features2
        # return x


class ResNetWithMask(nn.Module):
    def __init__(self, model_name='resnet200d', pretrained=False, n_class=19):
        super().__init__()
        # Naive CNN
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features

        self.stage1 = nn.Sequential(*list(self.model.children())[0:4])
        #         print(self.stage1)

        stage2 = list(self.model.children())[4]
        stage2 = list(stage2.children())
        self.stage2 = nn.Sequential(*stage2)

        stage3 = list(self.model.children())[5]
        stage3 = list(stage3.children())
        self.stage3 = nn.Sequential(*stage3)

        stage4 = list(self.model.children())[6]
        stage4 = list(stage4.children())
        self.stage4 = nn.Sequential(*stage4)

        stage5 = list(self.model.children())[7]
        stage5 = list(stage5.children())
        self.stage5 = nn.Sequential(*stage5)

        # Mask CNN
        self.model_mask = timm.create_model(model_name, pretrained=pretrained)

        self.stage1_mask = nn.Sequential(*list(self.model_mask.children())[0:4])
        #         print(self.stage1_mask)
        self.stage1_mask[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #         print(self.stage1_mask)
        self.conv1x1_sigmoid_shallow = nn.Sequential(
            nn.Conv2d(64, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),
            nn.Sigmoid()
        )

        stage2_mask = list(self.model_mask.children())[4]
        stage2_mask = list(stage2_mask.children())
        self.stage2_mask = nn.Sequential(*stage2_mask)

        stage3_mask = list(self.model_mask.children())[5]
        stage3_mask = list(stage3_mask.children())
        self.stage3_mask = nn.Sequential(*stage3_mask)

        stage4_mask = list(self.model_mask.children())[6]
        stage4_mask = list(stage4_mask.children())
        self.stage4_mask = nn.Sequential(*stage4_mask)

        stage5_mask = list(self.model_mask.children())[7]
        stage5_mask = list(stage5_mask.children())
        self.stage5_mask = nn.Sequential(*stage5_mask)

        self.conv1x1_sigmoid = nn.Sequential(
            nn.Conv2d(512, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),
            nn.Sigmoid()
        )

        # self.fc = nn.Linear(n_features, n_class, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # nn.Linear(n_features, n_features),
            nn.Dropout(0.1),  # 0.1待测试
            nn.Linear(n_features, n_class, bias=True)
        )

    def forward(self, x, x_mask):
        # x = self.model(x)
        # teacher and student method
        batch_size = x.size(0)
        # stage1
        x = self.stage1(x)
        x_mask = self.stage1_mask(x_mask)
        x_mask_a = self.conv1x1_sigmoid_shallow(x_mask)
        x = x * x_mask_a
        x_mask = x_mask + x
        # stage2
        x = self.stage2(x)
        x_mask = self.stage2_mask(x_mask)
        x = x + x_mask
        # stage3
        x = self.stage3(x)
        x_mask = self.stage3_mask(x_mask)
        x = x + x_mask
        # stage4
        x = self.stage4(x)
        x_mask = self.stage4_mask(x_mask)
        x = x + x_mask
        # stage5
        x = self.stage5(x)
        x_mask = self.stage5_mask(x_mask)
        # x_mask conv1x1+sigmoid
        x_mask = self.conv1x1_sigmoid(x_mask)

        features = x * x_mask

        pooled_features = self.avg_pool(features).view(batch_size, -1)
        x = self.fc(pooled_features)

        return x, features.view(features.shape[0], -1)
        # return x


class ResNetWithDrawMask(nn.Module):
    def __init__(self, model_name='resnet200d', pretrained=False, n_class=19):
        super().__init__()
        # Naive CNN
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features

        self.stage1 = nn.Sequential(*list(self.model.children())[0:4])
        #         print(self.stage1)

        stage2 = list(self.model.children())[4]
        stage2 = list(stage2.children())
        self.stage2 = nn.Sequential(*stage2)

        stage3 = list(self.model.children())[5]
        stage3 = list(stage3.children())
        self.stage3 = nn.Sequential(*stage3)

        stage4 = list(self.model.children())[6]
        stage4 = list(stage4.children())
        self.stage4 = nn.Sequential(*stage4)

        stage5 = list(self.model.children())[7]
        stage5 = list(stage5.children())
        self.stage5 = nn.Sequential(*stage5)

        # Mask CNN
        self.model_mask = timm.create_model(model_name, pretrained=pretrained)

        self.stage1_mask = nn.Sequential(*list(self.model_mask.children())[0:4])
        #         print(self.stage1_mask)
        #         self.stage1_mask[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #         print(self.stage1_mask)

        stage2_mask = list(self.model_mask.children())[4]
        stage2_mask = list(stage2_mask.children())
        self.stage2_mask = nn.Sequential(*stage2_mask)

        stage3_mask = list(self.model_mask.children())[5]
        stage3_mask = list(stage3_mask.children())
        self.stage3_mask = nn.Sequential(*stage3_mask)

        stage4_mask = list(self.model_mask.children())[6]
        stage4_mask = list(stage4_mask.children())
        self.stage4_mask = nn.Sequential(*stage4_mask)

        stage5_mask = list(self.model_mask.children())[7]
        stage5_mask = list(stage5_mask.children())
        self.stage5_mask = nn.Sequential(*stage5_mask)

        self.conv1x1_sigmoid = nn.Sequential(
            nn.Conv2d(512, 1, 1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'),
            nn.Sigmoid()
        )

        # self.fc = nn.Linear(n_features, n_class, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # nn.Linear(n_features, n_features),
            nn.Dropout(0.1),  # 0.1待测试
            nn.Linear(n_features, n_class, bias=True)
        )

    def forward(self, x, x_mask):
        # x = self.model(x)
        # teacher and student method
        batch_size = x.size(0)
        # stage1
        x = self.stage1(x)
        x_mask = self.stage1_mask(x_mask)
        x = x + x_mask
        # stage2
        x = self.stage2(x)
        x_mask = self.stage2_mask(x_mask)
        #         x = x + x_mask
        # stage3
        x = self.stage3(x)
        x_mask = self.stage3_mask(x_mask)
        #         x = x + x_mask
        # stage4
        x = self.stage4(x)
        x_mask = self.stage4_mask(x_mask)
        #         x = x + x_mask
        # stage5
        x = self.stage5(x)
        x_mask = self.stage5_mask(x_mask)
        # x_mask conv1x1+sigmoid
        x_mask = self.conv1x1_sigmoid(x_mask)

        features = x * x_mask

        pooled_features = self.avg_pool(features).view(batch_size, -1)
        x = self.fc(pooled_features)

        return x, features.view(features.shape[0], -1)
        # return x


class AdaptiveConcatPool2d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)), dim=1)


class Densenet(nn.Module):
    def __init__(self, model_name='densenet121', pretrained=False, n_class=19):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(n_features, n_class)
        # 提取中间层features
        self.feature = nn.Sequential(*list(self.model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(n_features, n_class)
        #         self.classifier = nn.Sequential(
        #             nn.Dropout(0.1), # 0.1待测试
        #             nn.Linear(n_features, n_class, bias=True)
        #         )
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512), Mish(), nn.BatchNorm1d(512), nn.Dropout(0.5), nn.Linear(512, n_class)
        )

    def forward(self, x):
        # x = self.model(x)
        # return x

        # teacher and student method
        batch_size = x.size(0)
        features = self.feature(x)
        pooled_features = self.avg_pool(features).view(batch_size, -1)
        x = self.classifier(pooled_features)
        return x, pooled_features


class ResNet_head(nn.Module):
    def __init__(self, model_name='resnet200d', pretrained=False, n_class=19):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.feature = nn.Sequential(*list(self.model.children())[:-2])
        self.avg_pool = AdaptiveConcatPool2d()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(n_features * 2),
            nn.Dropout(0.5),
            nn.Linear(n_features * 2, n_features, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.5),
            nn.Linear(n_features, n_class, bias=True)
        )

    def forward(self, x):
        # x = self.model(x)
        # teacher and student method
        batch_size = x.size(0)
        features = self.feature(x)
        pooled_features = self.avg_pool(features).view(batch_size, -1)
        x = self.fc(pooled_features)

        return x
        # return x


class ResNest(nn.Module):
    def __init__(self, model_name='resnest50d', pretrained=False, n_class=11):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(n_features, n_class)

        # 尝试此种fine-tune方式
        self.model.fc = nn.Sequential(
            nn.Dropout(0.1),  # 0.1待测试
            nn.Linear(n_features, n_class, bias=True)
        )

        # teacher and student method fine-tune
        # self.feature = nn.Sequential(list(self.model.children())[:-2])
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        # teacher and student method
        # batch_size = x.size(0)
        # features = self.feature(x)
        # pooled_features = self.avg_pool(features).view(batch_size, -1)
        # x = self.fc(pooled_features)
        return x
        # return features, pooled_features, x


class Inception_Resnet(nn.Module):
    def __init__(self, model_name='inception_resnet_v2', pretrained=False, n_class=11):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classif.in_features
        # teacher and student method fine-tune
        self.feature = nn.Sequential(*list(self.model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(n_features, n_class)
        # 尝试此种fine-tune方式
        self.classif = nn.Sequential(
            nn.Dropout(0.1),  # 0.1待测试
            nn.Linear(n_features, n_class, bias=True)
        )

    def forward(self, x):
        # x = self.model(x)
        # teacher and student method
        batch_size = x.size(0)
        features = self.feature(x)
        pooled_features = self.avg_pool(features).view(batch_size, -1)
        x = self.classif(pooled_features)
        return features, pooled_features, x


class Xception(nn.Module):
    def __init__(self, model_name='xception', pretrained=False, n_class=11):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b3', pretrained=False, n_class=11):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(n_features, n_class)
        # 提取中间层features
        self.feature = nn.Sequential(*list(self.model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(n_features, n_class)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),  # 0.1待测试
            nn.Linear(n_features, n_class, bias=True)
        )

        # self.model.classifier = nn.Sequential(
        #     nn.Dropout(0.1),
        #     #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
        #     nn.Linear(n_features, n_class, bias=True)
        # )

    def forward(self, x):
        # x = self.model(x)
        # return x

        # teacher and student method
        batch_size = x.size(0)
        features = self.feature(x)
        pooled_features = self.avg_pool(features).view(batch_size, -1)
        x = self.classifier(pooled_features)
        return features, pooled_features, x


if __name__ == '__main__':
    test_model = EfficientNet(model_name='tf_efficientnet_b7', pretrained=True)
    print(test_model)
    # test_data = torch.ones((1,3,224,2224)) # size is [batch, n_class]
    # print(test_model(test_data).size())
    # new_model = nn.Sequential(*list(test_model.modules())[:-3])
    # print(new_model)
    # print(list(test_model.named_modules())[-2:])
    # print(name, module)
