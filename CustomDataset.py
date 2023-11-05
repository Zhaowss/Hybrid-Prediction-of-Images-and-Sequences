import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # 初始化数据集
        self.data = pd.read_csv(csv_file)  # 从CSV文件加载数据
        self.transform = transform  # 用于图像数据的转换操作

    def __len__(self):
        # 返回数据集的长度，通常是数据样本的数量
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据集中指定索引的样本

        # 从数据中获取图像文件名和标签
        img_name = self.data.iloc[idx, -2]  # 图像文件名在倒数第二列
        label = self.data.iloc[idx, -1]  # 标签在最后一列

        # 构建完整的图像文件路径
        img_name = os.path.join("data/2D", str(label), str(img_name) + ".png")

        # 使用Pillow打开图像文件
        img = Image.open(img_name)

        if self.transform:
            # 如果指定了图像变换操作，应用变换到图像
            img = self.transform(img)

        # 构建一个样本字典，包括图像、数据和标签
        sample = {
            'image': img,  # 图像数据
            'data': torch.tensor(self.data.iloc[idx, :-2].values.astype('float64')).double(),  # 该行数据除了最后两列之外的数据，转换为Double类型
            'label': label - 1  # 标签，这里减去1是因为标签从0开始计数
        }

        return sample
