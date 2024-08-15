from os.path import join
from options.train_options import TrainOptions
from engine import Engine
import time

import torch.backends.cudnn as cudnn

from data import specdata
from torch.utils.data import DataLoader

import metrics as Metrics

opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log = True

# 加载测试数据集（此处更换测试数据集的路径）
# dataset_test = specdata.TestDataset('/mnt/data2/zwt/Project_closing/xiaomi/dataset/oppo_test/')
dataset_test = specdata.TestDataset('/mnt/data2/zwt/Project_closing/xiaomi/dataset/oppo_eyeglass/')
dataloader_test = DataLoader(dataset_test, 1, num_workers=opt.nThreads, shuffle=True, drop_last=False)

engine = Engine(opt, m_items=None)


# 开始测试并保存测试结果（此处更换测试结果保存路径）
engine.test(dataloader_test, savedir=join('./results', 'test_oppo_eyeglass_mynet01'))


