from os.path import join
from options.train_options import TrainOptions
import models
import torch.backends.cudnn as cudnn
from data import specdata
import torch
from torch.utils.data import DataLoader
import util
import time

opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log =True

# 加载测试数据集
dataset_test = specdata.TestDataset(opt.test_path)
dataloader_test = DataLoader(dataset_test,1, num_workers=opt.nThreads, shuffle=True, drop_last=False)

# 加载主程序
model = models.refectionModel(opt, m_items=None) # 加载模型文件

with torch.no_grad():
    for i, data in enumerate(dataloader_test):
        start = time.time()
        model.test(data, savedir=join(opt.save_path, opt.name))
        end = time.time()
        print("model infer time(s):", end - start)
        util.progress_bar(i, len(dataloader_test))
