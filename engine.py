import torch
# import util.util as util
import util as util
import models
import time
import os
from os.path import join


class Engine(object):
    def __init__(self, opt, m_items):
        self.opt = opt
        self.writer = None
        self.model = None
        self.best_val_loss = 1e6
        self.m_items = m_items

        self.__setup()

    # 启动模型
    def __setup(self):
        self.basedir = join('checkpoints', self.opt.name)  # 定义模型文件保存路径
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)

        opt = self.opt

        """Model"""
        self.model = models.refectionModel(opt, self.m_items)  # 加载模型文件
        if opt.resume:
            self.model.epoch += 1
        if not opt.no_log:  # 输出训练日志
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))

    def train(self, train_loader, **kwargs):
        print('\nEpoch: %d' % self.epoch)
        avg_meters = util.AverageMeters()
        opt = self.opt
        model = self.model

        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            iterations = self.iterations

            model.set_input(data, mode='train')
            model.optimize_parameters(**kwargs)

            errors = model.get_current_errors()
            avg_meters.update(errors)
            util.progress_bar(i, len(train_loader), str(avg_meters))

            if not opt.no_log:
                util.write_loss(self.writer, 'train', avg_meters, iterations)

            self.iterations += 1

        if not opt.no_log:
            if (self.epoch + 1) % opt.save_epoch_freq == 0:
                print('saving the model at epoch %d, iters %d' %
                      (self.epoch + 1, self.iterations))
                model.save()

            print('saving the latest model at the end of epoch %d, iters %d' %
                  (self.epoch + 1, self.iterations))
            model.save(label='latest')

            print('Time Taken: %d sec' %
                  (time.time() - epoch_start_time))

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                start = time.time()
                model.test(data, savedir=savedir, **kwargs)
                end = time.time()
                print("model infer time(s):", end - start)
                util.progress_bar(i, len(test_loader))

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e
