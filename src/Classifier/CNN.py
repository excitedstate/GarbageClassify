import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from cv2 import cv2
from src.Pretreatment import cv_show
import torch.utils.data as data
from src.Variables import *

# # hyper parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.con_1 = nn.Sequential(
            nn.Conv2d(  # -> (3, 400, 400)
                in_channels=3,  # # 原图像的层数 BGR为3层
                out_channels=8,  # # feature_map 的个数
                kernel_size=5,  # # 卷积核的大小
                stride=1,  # # 每次移动多少个像素
                padding=2  # # padding = (kernel_size - stride) // 2 = 2
            ), nn.ReLU(),  # -> (8, 400, 400)
            nn.MaxPool2d(kernel_size=4)  # -> (8, 100, 100)
        )
        self.con_2 = nn.Sequential(
            nn.Conv2d(  # -> (8, 200, 200)
                in_channels=8,  # # 类似的
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),  # -> (16, 200, 200)
            nn.MaxPool2d(kernel_size=4)  # -> (16, 25, 25)
        )
        self.hidden = nn.Linear(16 * 25 * 25, 88 * 3)
        self.out = nn.Linear(88 * 3, 88)

    def forward(self, x):
        x = self.con_2(self.con_1(x)).view(x.size(0), -1)  # # 卷积层
        return self.out(f.relu(self.hidden(x)))  # # 隐层和输出


class DataFrom:
    def __init__(self, is_train=True):
        self.is_train = is_train
        if is_train:
            self.images_mode = TRAIN_SET_PATH + "/train_images_%d.png"
            self.labels_mode = TRAIN_SET_PATH + "/train_labels_%d.npy"
        else:
            self.images_mode = TEST_SET_PATH + "/test_images_%d.png"
            self.labels_mode = TEST_SET_PATH + "/test_labels_%d.npy"

    def _read_image(self, start=0, end=1):
        ret_images = []
        ret_labels = []
        for i in range(start, end):
            img = cv2.imdecode(np.fromfile(self.images_mode % i, dtype=np.uint8), -1)
            ret_images.extend(np.hsplit(img, img.shape[1] // 400))
            ret_labels.extend(list(np.load(self.labels_mode % i)))
        return np.array(ret_images), np.array(ret_labels)  # # 这里改动了

    def data_loader(self, start=0, end=1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2):
        if self.is_train is False:
            return None
        ret_i, ret_l = self._read_image(start, end)
        train_x = torch.from_numpy(np.transpose(ret_i, (0, 3, 1, 2))).float()
        train_y = torch.from_numpy(ret_l).long()
        data_set = data.TensorDataset(train_x, train_y)
        return data.DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def test_loader(self, start=0, end=1):
        if self.is_train:
            return None
        ret_i, ret_l = self._read_image(start, end)
        return torch.from_numpy(np.transpose(ret_i, (0, 3, 1, 2))).float(), torch.from_numpy(ret_l).long()


if __name__ == "__main__":
    torch.manual_seed(1)  # # 随机数种子
    cnn = CNN()
    # # # data loader
    data_loader = DataFrom(is_train=True).data_loader()
    test_images, test_targets = DataFrom(is_train=False).test_loader()
    # # # optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    # # # training
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(data_loader):
            output = cnn(batch_x)
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                test_output = cnn(test_images)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_targets.data.numpy()).astype(int).sum()) / float(test_targets.size(0))
                print('Epoch: ', epoch, 'Step:', step, '| train loss: %.4f' % loss.data.numpy(),
                      '| test accuracy: %.2f' % accuracy)
    torch.save(cnn.state_dict(), "./gc.pkl")
    print('done')
