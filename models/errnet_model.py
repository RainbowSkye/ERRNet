import torch
from torch import nn
from collections import OrderedDict
import util
from util import tensor2im
import models.networks as networks
import models.losses as losses
from models import arch
from PIL import Image
import os
from os.path import join
from models.arch.errnet import FlowNet
import cv2
import dlib


class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        b, c, h, w = img.shape
        gradX = torch.zeros(b, 1, h, w, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(b, 1, h, w, dtype=img.dtype, device=img.device)

        gradx = (img[..., 1:, :] - img[..., :-1, :]).abs().sum(dim=1, keepdim=True)
        grady = (img[..., 1:] - img[..., :-1]).abs().sum(dim=1, keepdim=True)

        gradX[..., :-1, :] += gradx
        gradX[..., 1:, :] += gradx
        gradX[..., 1:-1, :] /= 2

        gradY[..., :-1] += grady
        gradY[..., 1:] += grady
        gradY[..., 1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge


class refectionModel:
    @staticmethod
    def name():
        return 'ERRNet'

    def __init__(self, opt, m_items):
        self.epoch = 0
        self.iterations = 0

        self.input = None
        self.map = None
        self.input_edge = None
        self.target = None
        self.mask = None
        self.data_name = None
        self.target_edge = None
        self.net_i = None

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.device = torch.device("cuda:{}".format(self.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        if m_items is not None:
            self.m_items = m_items.to(self.device)
        else:
            self.m_items = None

        in_channels = 3

        # 定义生成器网络结构
        self.net_i = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)  # 参数inet为选择使用的网络结构
        networks.init_weights(self.net_i, init_type='edsr')  # using default initialization as EDSR
        self.edge_map = EdgeMap(scale=1).to(self.device)

        if self.isTrain:
            # define loss functions
            self.loss_dic = {}

            pixel_loss = losses.MultipleLoss([nn.MSELoss(), losses.GradientLoss()], [0.2, 0.4])
            vgg_loss = losses.VGGLoss()

            disc_loss = losses.DiscLossRa()
            disc_loss.initialize(opt, self.Tensor)

            focal_loss = losses.BinaryFocalLoss()

            mask_loss = losses.MaskLoss()

            self.loss_dic['pixel'] = pixel_loss
            self.loss_dic['vgg'] = vgg_loss
            self.loss_dic['gan'] = disc_loss
            self.loss_dic['focal'] = focal_loss
            self.loss_dic['mask'] = mask_loss

            # Define discriminator
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self._init_optimizer([self.optimizer_D])

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net_i.parameters(), lr=opt.lr, betas=(0.9, 0.999),
                                                weight_decay=opt.wd)
            self._init_optimizer([self.optimizer_G])

            # Define FlowNet
            self.flownet = FlowNet().to(self.device)
            networks.init_weights(self.flownet, init_type='edsr')
            # initialize optimizers
            self.optimizer_F = torch.optim.Adam(self.flownet.parameters(),
                                                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)
            self._init_optimizer([self.optimizer_F])

        if opt.resume:
            self.load(self, opt.resume_epoch)
            # self.load_items(self. opt.resume_epoch)

        self.print_network()

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target = None
        map = None
        data_name = None
        mask = None
        mode = mode.lower()     # 将字符串转换为小写形式
        if mode == 'train':
            input, target, mask, map = data['input'], data['target'], data['mask'], data['map']
        elif mode == 'test':
            input, mask, data_name = data['input'], data['mask'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)

        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])

            if target is not None:
                target = target.to(device=self.gpu_ids[0])
            if mask is not None:
                mask = mask.to(device=self.gpu_ids[0])
            if map is not None:
                map = map.to(device=self.gpu_ids[0])

        self.input = input
        self.map = map
        self.input_edge = self.edge_map(self.input)
        self.target = target
        self.mask = mask
        self.data_name = data_name

        if target is not None:
            self.target_edge = self.edge_map(self.target)

    def test(self, data, savedir=None):
        self.net_i.eval()
        self.set_input(data, 'test')

        with torch.no_grad():
            self.forward(train=False)
            if self.data_name is not None and savedir is not None:
                os.makedirs(join(savedir), exist_ok=True)
                for _input, result, fn in zip(self.input, self.results[2],self.data_name):
                    _input = tensor2im(_input)
                    result = tensor2im(result)
                    Image.fromarray(result).save(join(savedir, '{}_result.png'.format(fn)))
                    Image.fromarray(_input).save(join(savedir, '{}_input.png'.format(fn)))

    def save(self, label=None):
        epoch = self.epoch
        iterations = self.iterations

        if label is None:
            model_name = os.path.join(self.save_dir, self.name() + '_%03d_%08d.pt' % ((epoch), (iterations)))
        else:
            model_name = os.path.join(self.save_dir, self.name() + '_' + label + '.pt')
        if label is None:
            items_name = os.path.join(self.save_dir, self.name() + '_items_%03d_%08d.pt' % ((epoch), (iterations)))
        else:
            items_name = os.path.join(self.save_dir, self.name() + '_items_' + label + '.pt')

        torch.save(self.state_dict(), model_name)
        torch.save(self.m_items.cpu(), items_name)


    def backward_F(self):
        for p in self.netD.parameters():
            p.requires_grad = False

        self.loss_F = 0
        self.landmark_loss = 0

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

        face = self.target_t
        flip_face_tensor = face[:, :, :, torch.arange(face.size(3) - 1, -1, -1).long()]
        self.flowmap = self.flownet.forward(face, flip_face_tensor)

        for i in range(face.shape[0]):

            face1 = tensor2im(face[i])
            flip_face = cv2.flip(face1, 1)

            face_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
            flip_gray = cv2.cvtColor(flip_face, cv2.COLOR_BGR2GRAY)

            # 对gt进行眼部分割
            faces = detector(face_gray, 1)
            flip_faces = detector(flip_gray, 1)
            for fa, flip_fa in zip(faces, flip_faces):
                keypoint = predictor(face1, fa)
                flip_keypoint = predictor(flip_face, flip_fa)
                for key, flip_key in zip(keypoint.parts(), flip_keypoint.parts()):
                    position = (key.x, key.y)
                    cv2.circle(face1, position, 2, (0, 0, 255), -1)
                    flip_position = (flip_key.x, flip_key.y)
                    cv2.circle(flip_face, flip_position, 2, (0, 0, 255), -1)

                # count loss
                for key, flip_key in zip(keypoint.parts(), flip_keypoint.parts()):
                    self.landmark_loss += abs(self.flowmap[i][0][flip_key.y - 1][flip_key.x - 1] - key.x) + abs(
                        self.flowmap[i][1][flip_key.y - 1][flip_key.x - 1] - key.y)

        self.loss_F += self.landmark_loss * 10.0

        # 求self.flowmap在x和y方向的梯度
        gradxx = self.flowmap[..., 1:, :] - self.flowmap[..., :-1, :]
        gradxy = self.flowmap[..., 1:] - self.flowmap[..., :-1]
        gradyx = self.flowmap[..., 1:, :] - self.flowmap[..., :-1, :]
        gradyy = self.flowmap[..., 1:] - self.flowmap[..., :-1]
        zero_tensor = torch.zeros(self.flowmap[0].shape).to(self.flowmap[0].device)
        self.TV_loss = nn.L1Loss(gradxx, zero_tensor) + nn.L1Loss(gradxy, zero_tensor) + nn.L1Loss(gradyx, zero_tensor) + nn.L1Loss(gradyy, zero_tensor)

        self.loss_F += self.TV_loss * 1.0

        self.loss_F.backward()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(self.netD, self.input, self.results[2], self.target)

        (self.loss_D * self.opt.lambda_gan).backward(retain_graph=True)

    def backward_G(self):
        # Make it a tiny bit faster
        for p in self.netD.parameters():
            p.requires_grad = False

        self.loss_G = 0
        self.loss_G_GAN = None
        self.loss_focal = None
        self.loss_mask = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(self.netD, self.input, self.results[2], self.target)
            self.loss_G += self.loss_G_GAN * self.opt.lambda_gan

        if self.opt.lambda_vgg > 0:
            self.loss_vgg = self.loss_dic['vgg'](self.results[2], self.target)
            self.loss_G += self.loss_vgg * self.opt.lambda_vgg

        if self.opt.lambda_flownet > 0:
            flowmap = self.flownet.forward(self.output_i, self.target_t)
            self.symmetry_loss = 0

            self.loss_G += self.symmetry_loss * self.opt.lambda_flownet

        self.loss_pixel = self.loss_dic['pixel'](self.results[2], self.target)
        self.loss_G += self.loss_pixel

        self.loss_coarse_pixel = self.loss_dic['pixel'](self.results[0], self.target) + self.loss_dic['pixel'](self.results[1], self.target)
        self.loss_coarse_pixel += (self.loss_dic['vgg'](self.results[0], self.target) + self.loss_dic['vgg'](self.results[1], self.target))  * self.opt.lambda_vgg
        self.loss_G += self.loss_coarse_pixel * self.opt.lambda_pixel

        # focal loss
        self.loss_focal = self.loss_dic['focal'](self.detect, self.mask)
        self.loss_G += self.loss_focal * self.opt.lambda_focal

        # mask loss
        self.loss_mask = self.loss_dic['mask'](self.weight_mask, self.map)
        self.loss_G += self.loss_mask * self.opt.lambda_mask

        self.loss_G += self.compactness_loss * 0.1 + self.separateness_loss * 0.1

        self.loss_G.backward()

    def forward(self, train=True):

        input_i = self.input

        # if not train:
        #     flops, params = profile(self.net_i, inputs=(input_i, self.m_items,False))
        #     print("flops:", flops)
        #     print("params:", params)

        if not self.m_items.is_cuda:
            self.m_items = self.m_items.to(self.device)
        output = self.net_i(input_i, self.m_items, train)
        self.results = [output['removal'],output['inpainting'],output['result']]
        self.detect = output['detect']
        self.weight_mask = output['weight_mask']
        self.compactness_loss = output['compactness_loss']
        self.separateness_loss = output['separateness_loss']
        self.m_items = output['m_items']

        self.detection = (self.detect.detach() > 0.5).type(torch.float32)

        return self.results

    def optimize_parameters(self):
        self.net_i.train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        if self.opt.lambda_flownet == -1:
            self.optimizer_F.zero_grad()
            self.backward_F()
            self.optimizer_F.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_pixel is not None:
            ret_errors['IPixel'] = self.loss_pixel.item()
        if self.loss_vgg is not None:
            ret_errors['VGG'] = self.loss_vgg.item()

        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()

        if self.compactness_loss and self.separateness_loss:
            ret_errors['compactness_loss'] = self.compactness_loss.item()
            ret_errors['separateness_loss'] = self.separateness_loss.item()

        if self.loss_focal is not None:
            ret_errors['Focal'] = self.loss_focal.item()

        if self.loss_mask is not None:
            ret_errors['Weight_mask'] = self.loss_mask.item()
        return ret_errors

    @staticmethod
    def load(model, resume_epoch=None):
        model_path = util.get_model_list(model.save_dir, model.name(), epoch=resume_epoch)
        items_path = os.path.join(model.save_dir, model.name() + '_items'+'_latest.pt')
        model.m_items = torch.load(items_path)
        state_dict = torch.load(model_path)
        model.epoch = state_dict['epoch']
        model.iterations = state_dict['iterations']
        model.net_i.load_state_dict(state_dict['icnn'])
        if model.isTrain:
            model.optimizer_G.load_state_dict(state_dict['opt_g'])

        if model.isTrain:
            if 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(state_dict['netD'])
                model.optimizer_D.load_state_dict(state_dict['opt_d'])

        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': self.net_i.state_dict(),
            'opt_g': self.optimizer_G.state_dict(),
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': self.netD.state_dict(),
            })

        return state_dict
