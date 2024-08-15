import argparse
import models
import util
import os
import torch
import numpy as np
import random

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default=None, help='name of the experiment. It '
                                                                        'decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='1,7', help='gpu ids: e.g. 0  0,1,2, '
                                                                            '0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are'
                                                                                              ' saved here')
        self.parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--resume_epoch', '-re', type=int, default=None, help='checkpoint to use. '
                                                                                       '(default: latest')
        self.parser.add_argument('--seed', type=int, default=2018, help='random seed to use. Default=2018')
        self.parser.add_argument('--inet', type=str, default='errnet', help='chooses which architecture to use.')
        # for setting input
        self.parser.add_argument('--train_path', default='./Face_Specular/Train/', type=str, help='the '
                                                                                                  'path of train data')
        self.parser.add_argument('--test_path', default='./Face_Specular/Test/', type=str, help='the path of test data')
        self.parser.add_argument('--fliplr', default=0.5, type=float, help='the rate to flip the image left-right')
        self.parser.add_argument('--flipud', default=0.5, type=float, help='the rate to flip the image up-down')
        self.parser.add_argument('--noise', default=True, type=bool, help='add the noise to the image')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')

        self.parser.add_argument('--save_path', default='./results/', type=str, help='the path of results')
        self.parser.add_argument('--no-log', action='store_true', help='disable tf logger?')
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.opt.seed)
        np.random.seed(self.opt.seed)  # seed for every module
        random.seed(self.opt.seed)

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk02
        self.opt.name = self.opt.name or '_'.join([self.opt.model])
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return self.opt
