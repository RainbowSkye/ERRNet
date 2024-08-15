import torch
import os.path
from os.path import join
import numpy as np
import cv2
import math

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, opt, datadir, path1='specular', path2='gt'):
        super(TrainDataset, self).__init__()
        self.opt = opt
        self.datadir = datadir
        self.path1 = path1
        self.path2 = path2
        self.fns1 = sorted(os.listdir(join(datadir, path1)))
        self.fns2 = self.fns1
        # self.fns2 = sorted(os.listdir(join(datadir, path2)))
        print('Load {} items in {} ...'.format(len(self.fns1), datadir))

    def __getitem__(self, index):
        fn1 = self.fns1[index]
        fn2 = self.fns2[index]
        m_img = cv2.imread(join(self.datadir, self.path1, fn1))  # 反光图像
        t_img = cv2.imread(join(self.datadir, self.path2, fn2))  # GT
        if np.random.rand() < self.opt.fliplr:
            t_img = cv2.flip(t_img, 1)  # 水平翻转图像
            m_img = cv2.flip(m_img, 1)
        if np.random.rand() < self.opt.flipud:
            t_img = cv2.flip(t_img, 0)  # 垂直翻转图像
            m_img = cv2.flip(m_img, 0)

        # 确定图片下采样后的大小
        if m_img.shape[0] < m_img.shape[1]:
            size = (int(256 * m_img.shape[1] / m_img.shape[0]), 256)
        else:
            size = (256, int(256 * m_img.shape[0] / m_img.shape[1]))

        if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
            scale = int(math.log2(min(m_img.shape[0] / size[1], m_img.shape[1] / size[0])))
            for i in range(0, scale):
                m_img = cv2.pyrDown(m_img)
                t_img = cv2.pyrDown(t_img)
            if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]) or not (
                    t_img.shape[0] == size[1] and t_img.shape[1] == size[0]):
                m_img = cv2.resize(m_img, size, cv2.INTER_AREA)
                t_img = cv2.resize(t_img, size, cv2.INTER_AREA)

        t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)  # OpenCV中图像以BGR颜色通道顺序存储, 该代码将BGR图像转换为RGB图像
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)

        M = np.transpose(np.float32(m_img) / 255.0, (2, 0, 1))
        T = np.transpose(np.float32(t_img) / 255.0, (2, 0, 1))
        delta = M - T
        mask = 0.3 * delta[0] + 0.59 * delta[1] + 0.11 * delta[2]  # 将彩色图像转换为灰度图像,0.3、0.59、0.11是考虑到人眼对不同颜色通道的感知差异而设置的权重
        mask = np.float32(mask > 0.707 * mask.max())
        if self.opt.noise:
            M = M + np.random.normal(0, 2 / 255.0, M.shape).astype(np.float32)
        data = {'input': M, 'target': T, 'fn': fn1[:-4], 'mask': mask, 'map': delta}
        return data

    def __len__(self):
        return len(self.fns1)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, datadir):
        super(TestDataset, self).__init__()
        self.datadir = datadir
        self.fns = sorted(os.listdir(datadir))
        print('Load {} items in {} ...'.format(len(self.fns), datadir))

    def __getitem__(self, index):
        fn = self.fns[index]
        m_img = cv2.imread(join(self.datadir, fn))

        # if m_img.shape[0] < m_img.shape[1]:
        #     if int(256*m_img.shape[1]/m_img.shape[0]) % 2 == 0:
        #         mmm = int(256*m_img.shape[1]/m_img.shape[0])
        #     else:
        #         mmm = int(256*m_img.shape[1]/m_img.shape[0]) + 3
        #     size = (mmm,256)
        # else:
        #     if int(256*m_img.shape[0]/m_img.shape[1]) % 2 == 0:
        #         mmm = int(256*m_img.shape[0]/m_img.shape[1])
        #     else:
        #         mmm = int(256*m_img.shape[0]/m_img.shape[1]) + 3
        #     size = (256,mmm)
        if m_img.shape[0] < m_img.shape[1]:
            temp = int(256 * m_img.shape[1] / m_img.shape[0])
            size = (temp - temp % 16, 256)
        else:
            temp = int(256 * m_img.shape[0] / m_img.shape[1])
            size = (256, temp - temp % 16)

        if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
            scale = int(math.log2(min(m_img.shape[0] / size[1], m_img.shape[1] / size[0])))
            for i in range(0, scale):
                m_img = cv2.pyrDown(m_img)
            if not (m_img.shape[0] == size[1] and m_img.shape[1] == size[0]):
                m_img = cv2.resize(m_img, size, cv2.INTER_AREA)
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
        M = np.transpose(np.float32(m_img) / 255.0, (2, 0, 1))
        data = {'input': M, 'target_t': torch.zeros([1, 0]), 'fn': fn[:-4], 'mask': torch.zeros([1, 0])}
        return data

    def __len__(self):
        return len(self.fns)
