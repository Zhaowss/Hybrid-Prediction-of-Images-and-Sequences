import torch
import torch.nn as nn

class CNN1DModel(nn.Module):
    def __init__(self, num_out):
        super(CNN1DModel, self).__init__()

        # 1D卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3)
        # in_channels: 输入通道数，对于1维数据，通常为1（每个样本的特征通道数）
        # out_channels: 卷积核的数量，控制特征提取的多少
        # kernel_size: 卷积核的大小，用于在输入数据上提取特征

        # 最大池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        # kernel_size: 池化窗口的大小，用于下采样特征

        # 全连接层
        self.fc1 = nn.Linear(640, 128)
        # 输入维度为 640，这是根据卷积层和池化层的输出维度确定的
        # 输出维度为 128，这是全连接层的隐藏单元数量，可以根据需求调整

        # 输出层
        self.fc2 = nn.Linear(128, num_out)
        # 输入维度为 128，与前面全连接层的输出维度相匹配
        # 输出维度为 num_out，通常对应于分类的类别数量

    def forward(self, x):
        # 在第二个维度上增加一个通道维度
        x = x.unsqueeze(1)
        # x 的形状从 (batch_size, sequence_length, input_size) 变为 (batch_size, 1, sequence_length, input_size)
        # 这是因为卷积操作期望一个4维输入

        # 进行卷积操作，激活函数是ReLU
        x = self.conv1(x)
        x = torch.relu(x)

        # 最大池化，用于下采样特征
        x = self.pool(x)

        # 展平数据，将3维数据转换为2维，以便进行全连接操作
        x = x.view(x.size(0), -1)

        # 进行全连接操作，激活函数是ReLU
        x = torch.relu(self.fc1(x))

        # 最后的全连接层，用于输出分类结果
        x = self.fc2(x)

        return x
