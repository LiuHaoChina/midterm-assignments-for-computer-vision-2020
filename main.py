import sys
import math
import torch
import torch.nn.functional as F
import time
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import pandas as pd
import os
import cv2
import numpy as np


# 路径转换
def convert_path(path: str) -> str:
    seps = r'\/'
    sep_other = seps.replace(os.sep, '')
    return path.replace(sep_other, os.sep) if sep_other in path else path


windows = sys.platform.startswith('win')
# 本地数据集存放路径
root_path = '/Users/monsterliu/Desktop/计算机视觉/homework'
if windows:
    root_path = 'F:\AliTianChi\zuoye\MURA-v1.1'

data_train_img_csv_path = root_path + os.sep + 'MURA-v1.1' + os.sep + 'train_image_paths.csv'
data_train_label_csv_path = root_path + os.sep + 'MURA-v1.1' + os.sep + 'train_labeled_studies.csv'

data_test_img_cvs_path = root_path + os.sep + 'MURA-v1.1' + os.sep + 'valid_image_paths.csv'
data_test_label_cvs_path = root_path + os.sep + 'MURA-v1.1' + os.sep + 'valid_labeled_studies.csv'


# 读取本地原始数据集，并构造符合自己需求的dataset格式
class LocalXRayDataSet(Dataset):
    def __init__(self, local_img_csv_path, local_label_csv, loader=default_loader):
        imgs = pd.read_csv(local_img_csv_path, names=['img_path'])
        self.imgs_path = imgs['img_path'].values
        self.len = imgs.__len__()

        img_label = pd.read_csv(local_label_csv, names=['dir_path', 'label'])
        # 转换路径
        if sys.platform.startswith('win'):
            for i in range(img_label.__len__()):
                img_label['dir_path'].values[i] = convert_path(img_label['dir_path'].values[i])
        self.dir_path = img_label['dir_path'].values
        self.label = img_label['label'].values

        self.map = img_label.set_index(['dir_path'])['label'].to_dict()
        # self.len = img_label.__len__()

    # 迭代时会调用这个方法来依据下标获取单个文件夹内的数据
    def __getitem__(self, index):
        img_path = root_path + os.sep + self.imgs_path[index - 1]
        img_label = self.map[os.path.relpath(os.path.dirname(img_path), root_path) + os.sep]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 先无脑resize了 todo spp层
        img = cv2.resize(img, dsize=(128, 128))
        # print(img.shape)
        # img = np.expand_dims(img, 0)
        # print(img.shape)
        img = img.astype(np.float32) / 255
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        # if use_gpu:
        #     img = img.cuda()
        #     img_label = img_label.cuda()
        return img, img_label
        # dir_full_path = root_path + '/' + self.dir_path[index - 1]
        # # 读取一个文件夹下的所有图像
        # file_names = os.listdir(dir_full_path)
        # images = []
        # for img_name in file_names:
        #     image = cv2.imread(dir_full_path + img_name)
        #     image = torch.from_numpy(image)
        #     images.append(image)
        # # 返回 标签值、本次含有的图像数量
        # return self.label, file_names.__len__(), images

    # 迭代时会依据这个方法来判断每次迭代中数据分多少批次
    def __len__(self):
        return self.len


# 调用方法获取训练数据集
data_train = LocalXRayDataSet(local_img_csv_path=data_train_img_csv_path, local_label_csv=data_train_label_csv_path)
data_test = LocalXRayDataSet(local_img_csv_path=data_test_img_cvs_path, local_label_csv=data_test_label_cvs_path)

'''
    定义超参数
    batch_size以及num_workers的设置
    这里看电脑性能如何，cpu好，内存大的话，batch_size和num_workers考虑设置的可以大一些
    本质上这里和训练平台无关，只是本人所使用平台具有较大的区分度，就这样写了
    本人的windows电脑性能略好一些，故而这块设置的较大一点
'''
use_gpu = torch.cuda.is_available()
print(f'是否使用GPU：{use_gpu}')
learning_rate = 1e-3
num_epochs = 100
if sys.platform.startswith('win'):
    num_workers = 2
    batch_size = 8
else:
    num_workers = 2
    batch_size = 8
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=num_workers)

# # 构建SPP层(空间金字塔池化层)
# class SPPLayer(torch.nn.Module):
#     def __init__(self, num_levels, pool_type='max_pool'):
#         super(SPPLayer, self).__init__()
#
#         self.num_levels = num_levels
#         self.pool_type = pool_type
#
#     def forward(self, x):
#         num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
#         for i in range(self.num_levels):
#             level = i + 1
#             kernel_size = (math.ceil(h / level), math.ceil(w / level))
#             stride = (math.ceil(h / level), math.ceil(w / level))
#             pooling = (
#                 math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))
#
#             # 选择池化方式
#             if self.pool_type == 'max_pool':
#                 tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
#             else:
#                 tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
#
#             # 展开、拼接
#             if i == 0:
#                 x_flatten = tensor.view(num, -1)
#             else:
#                 x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
#         return x_flatten


"""
    定义模型
    先自己定一个简单的，后面再改成ResNet这种复杂的试试
"""


class MyModule(nn.Module):
    def __init__(self, n_class):
        super(MyModule, self).__init__()
        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        # 定义三层全连接层
        self.fc = nn.Sequential(
            nn.Linear(14400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        # 在此构建神经网络的整体模型层次，将会按此处顺序执行下去
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 构造模型，定义loss(损失函数)以及optimizer(优化算法)，这是个2分类问题，所以 n_class = 2
model = MyModule(2)
if use_gpu:
    model = model.cuda()
# 使用交叉熵以及SGD优化
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    start_time = time.time_ns()
    # 迭代训练
    for epoch in range(num_epochs):
        epoch_start_time = time.time_ns()
        print('*' * 10)
        print(f'epoch {epoch + 1}')
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
            img, label = data
            # img = img.view(img.size(0), -1)
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            # 前向传播
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item()
            _, pred = torch.max(out, 1)
            running_acc += (pred == label).float().mean()

            # 后向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 300 == 0:
                process = round(i * 100 * batch_size / 36808, 2)
                print(
                    f'[{epoch + 1}/{num_epochs}] 单次训练进度：{process}%, Loss: {running_loss / i:.6f}, Acc: {running_acc / i:.6f}')
        # 完成一次整体的迭代训练，查看训练迭代效果
        print(f'Finish {epoch + 1} epoch, Loss: {running_loss / i:.6f}, Acc: {running_acc / i:.6f}')
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for data in test_loader:
            img, label = data
            # img = img.view(img.size(0), -1)
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            with torch.no_grad():
                out = model(img)
                loss = criterion(out, label)
            eval_loss += loss.item()
            _, pred = torch.max(out, 1)
            eval_acc += (pred == label).float().mean()
        # 完成一次整体的迭代训练，查看在测试集合上的表现
        print(f'Test Loss: {eval_loss / len(test_loader):.6f}, Acc: {eval_acc / len(test_loader):.6f}\n')
        epoch_cost_time = time.time_ns() - epoch_start_time
        epoch_cost_time = round(epoch_cost_time / 1000000000, 4)
        print(f'耗时：{epoch_cost_time}s\n')
    cost_time = time.time_ns() - start_time
    cost_time = round(cost_time/1000000000, 4)
    print(f'共计耗时{cost_time}s')
    # 保存模型
    torch.save(model.state_dict(), './neural_network.pth')
