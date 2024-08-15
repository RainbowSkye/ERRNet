# import util.util as util
import util as util
from data import specdata
from torch.utils.data import DataLoader
from os.path import join
from options.train_options import TrainOptions
from engine import Engine
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F


# 设置学习率函数用于更新学习率
def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


# 加载训练参数
opt = TrainOptions().parse()

cudnn.benchmark = True

# 加载训练数据集（此处更换训练数据集的路径）（同时训练数据集划分为Specular和GT两个子目录）
dataset_train = specdata.TrainDataset(opt, './dataset/train', path1='specular', path2='gt')
dataloader_train = DataLoader(dataset_train, opt.batchSize, num_workers=opt.nThreads, shuffle=True, drop_last=False)
# 加载测试数据集（此处更换测试数据集的路径）
dataset_test = specdata.TestDataset('./dataset/val')
dataloader_test = DataLoader(dataset_test, 1, num_workers=opt.nThreads, shuffle=True, drop_last=False)

m_items = F.normalize(torch.rand((512, 512), dtype=torch.float), dim=1)  # Initialize the memory items

# 加载主程序
engine = Engine(opt, m_items)

# 初始化对抗损失权重为0
engine.model.opt.lambda_gan = 0
# 初始化学习率为1e-4
lr = 1e-4

while engine.epoch < opt.nEpochs:
    if engine.epoch >= 20:
        engine.model.opt.lambda_gan = 0.01  # 20个epoch之后使用对抗损失

    if engine.epoch >= 40:
        engine.model.opt.lambda_flownet = 0.01

    if (engine.epoch + 1) % 5 == 0:  # 每五个epoch更新一次学习率
        lr_now = max(1e-5, lr * 0.8 ** ((engine.epoch + 1) / 5))
        set_learning_rate(lr_now)
    if True:
        print("training ...")
        engine.train(dataloader_train)  # 开始训练
        engine.epoch += 1
        if engine.epoch % 5 == 0:  # 每五个epoch输出一次测试结果
            engine.test(dataloader_test, savedir=join('./results', opt.name))
