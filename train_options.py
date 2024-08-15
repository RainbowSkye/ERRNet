from base_option import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)        
        # for displays
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at '
                                                                                 'the end of epochs')

        # for training (Note: in train.py, we mannually tune the training protocol, but you can also use following
        # setting by modifying the code in specularitynet_model.py)
        self.parser.add_argument('--nEpochs', '-n', type=int, default=80, help='# of epochs to run')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--wd', type=float, default=0, help='weight decay for adam')
        self.parser.add_argument('--batchSize', '-b', type=int, default=1, help='input batch size')
        
        # loss weight
        self.parser.add_argument('--lambda_gan', type=float, default=0.01, help='weight for gan loss')
        self.parser.add_argument('--lambda_flownet', type=float, default=0, help='weight for symmetry loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=0.1, help='weight for vgg loss')
        self.parser.add_argument('--lambda_pixel', type=float, default=0.5, help='weight for pixel loss')
        self.parser.add_argument('--lambda_focal', type=float, default=1.0, help='weight for focal loss')
        self.parser.add_argument('--lambda_mask', type=float, default=10.0, help='weight for mask loss')
        self.isTrain = True
