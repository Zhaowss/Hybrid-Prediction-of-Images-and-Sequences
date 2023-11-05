import torch
import torch.nn as nn
import torchvision.models as models
from resnet import resnet50  # 导入自定义的resnet50模型
from My1Dmodel import CNN1DModel  # 导入自定义的1D CNN模型


class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()

        # 创建ResNet50骨干网络
        self.resnet = resnet50(pretrained=False)  # 使用自定义的resnet50模型，可以是PyTorch官方的也可以是自己实现的
        self.myoneDnet = CNN1DModel(num_out=64)  # 创建自定义的1D CNN模型

        # 去掉ResNet的原始全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children()))

        # 添加自定义分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，将特征图池化成固定大小
            nn.Flatten(),  # 展平操作，将池化后的特征图展开成一维向量
            nn.Linear(2048, 128)  # 全连接层，这里假设有128个隐藏单元
        )

        self.Convert = nn.Linear(192, num_classes)  # 自定义的全连接层，用于最终的分类

    def forward(self, x1, x2):
        # 通过ResNet的骨干网络去获取我们的图像的特征
        x1 = self.resnet(x1)  # 使用ResNet进行特征提取
        x1 = self.classifier(x1)  # 使用自定义的分类层，将其特征送入FC层得到1D的数据

        x2 = self.myoneDnet(x2.float())  # 使用自定义的1D CNN模型，将我们的序列数据输入1DCNN在送入FC
        x = torch.cat([x1, x2], dim=1)  # 将两个模型的输出连接在一起，dim=1表示在第二个维度上拼接

        # 通过自定义的全连接层进行最终的分类
        x = self.Convert(x)

        return x
