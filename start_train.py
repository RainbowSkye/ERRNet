from data import specdata
from torch.utils.data import DataLoader
from os.path import join
from options.train_options import TrainOptions
import util
import models
import os
import time
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F


# 设置学习率函数用于更新学习率
def set_learning_rate(lr):
    for optimizer in model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


def test(test_loader, savedir=None):
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            model.test(data, savedir=savedir)
            util.progress_bar(i, len(test_loader))


# 加载训练参数
opt = TrainOptions().parse()

cudnn.benchmark = True

# 加载训练数据集（同时训练数据集划分为Specular和GT两个子目录）
dataset_train = specdata.TrainDataset(opt, opt.train_path, path1='Specular', path2='GT')
dataloader_train = DataLoader(dataset_train, opt.batchSize, num_workers=opt.nThreads, shuffle=True, drop_last=False)
# 加载测试数据集
dataset_test = specdata.TestDataset(opt.test_path)
dataloader_test = DataLoader(dataset_test, 1, num_workers=opt.nThreads, shuffle=True, drop_last=False)

m_items = F.normalize(torch.rand((512, 512), dtype=torch.float), dim=1)     # Initialize the memory items

# 加载主程序
model = models.refectionModel(opt, m_items)     # 加载模型文件

# 初始化对抗损失权重为0
model.opt.lambda_gan = 0
# 初始化学习率为1e-4
lr = 1e-4

# Report the training process
basedir = join('checkpoints', opt.name)     # 定义模型文件保存路径
if not os.path.exists(basedir):
    os.mkdir(basedir)

writer = util.get_summary_writer(os.path.join(basedir, 'logs'))

for epoch in range(opt.nEpochs):

    if epoch >= 20:
        model.opt.lambda_gan = 0.01     # 20个epoch之后使用对抗损失

    if (epoch+1) % 5 == 0:    # 每五个epoch更新一次学习率
        lr_now = max(1e-5, lr*0.8**((epoch+1)/5))
        set_learning_rate(lr_now)

    avg_meters = util.AverageMeters()
    epoch_start_time = time.time()
    for i, data in enumerate(dataloader_train):

        iterations = model.iterations

        model.set_input(data, mode='train')
        model.optimize_parameters()

        errors = model.get_current_errors()
        avg_meters.update(errors)
        util.progress_bar(i, len(dataloader_train), str(avg_meters))

        if not opt.no_log:
            util.write_loss(writer, 'train', avg_meters, iterations)

        model.iterations += 1

    if not opt.no_log:
        if (model.epoch + 1) % opt.save_epoch_freq == 0:
            print('saving the model at epoch %d, iters %d' % (model.epoch + 1, model.iterations))
            model.save()

        print('saving the latest model at the end of epoch %d, iters %d' % (model.epoch + 1, model.iterations))
        model.save(label='latest')

        print('Time Taken: %d sec' %
              (time.time() - epoch_start_time))

    if epoch % 1 == 0:
        test(dataloader_test, savedir=join(opt.save_path, opt.name))