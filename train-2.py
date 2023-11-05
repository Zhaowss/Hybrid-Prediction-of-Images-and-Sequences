import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from CustomDataset import  CustomDataset
from mynetwork import CustomResNet


def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 定义模型image的预处理部分，包含resize和和归一化的部分
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # 设置1D数据读取的地址
    csv_path = "data/1D/total.csv" # flower data set path
    assert os.path.exists(csv_path), "{} path does not exist.".format(csv_path)
    # 加载训练集和测试集
    train_dataset = CustomDataset(csv_file=csv_path,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    validate_dataset =  CustomDataset(csv_file=csv_path,
                                         transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=2, shuffle=False,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # 设置batch size
    batch_size = 2
    # 设置线程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)



    print("using {} images for training".format(train_num))
    # 获取我们自定义的网络
    net = CustomResNet(9)
    # 将网络指认到我们自己的设备上面
    net.to(device)
    # 定义我们自己的损失函数，这里采用卡放损失
    # define loss function
    loss_function = nn.CrossEntropyLoss()
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    # 定义优化器
    optimizer = optim.Adam(params, lr=0.0001)
    # 设置训练的次数
    epochs = 3
    best_acc = 0.0
    # 设置模型保存的地址
    save_path = './Mymodel.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, mydata in enumerate(train_bar):
            # 获取我们的定义的dataset返回的图像数据，1D的序列的数据，以及我们的标签的数据
            image, data,label = mydata['image'],mydata['data'],mydata['label']
            optimizer.zero_grad()
            # 将数据输入我们定义的网络中
            logits = net(image.to(device), data.to(device))
            # 计算损失函数

            loss = loss_function(logits, label.to(device))
            # 损失反响传播
            loss.backward()
            # 优化器迭代优化参数
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # 验证集验证部分
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images,val_OneDim_data, val_labels = val_data['image'],val_data['data'],val_data['label']
                outputs = net(val_images.to(device),val_OneDim_data.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()