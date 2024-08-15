import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
import pdb
###############################################################################
# Functions
###############################################################################
def compute_gradient(img):
    gradx = img[..., 1:, :]-img[..., :-1, :]
    grady = img[..., 1:]-img[..., :-1]
    return gradx,grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target) 
        
        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i + 1) in indices:
                out.append(X)
        return out


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()        
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        
        return loss


class MaskLoss(nn.Module):
    def __init__(self, t = 0.1):
        super(MaskLoss, self).__init__()
        self.t = t

    def convert_L(self, img):
        img_L = (0.114 * img[0, 0, :, :] + 0.587 * img[0, 1, :, :] + \
                 0.299 * img[0, 2, :, :]).unsqueeze(0).unsqueeze(0)
        return img_L

    def l1_loss(self, input,output):
        return torch.mean(torch.abs(input - output))

    def forward(self, weight_mask, map):
        map_L = self.convert_L(map)
        map_gt = torch.zeros(map_L.shape).to(map.device)
        map_gt[0, 0, map_L[0,0,:,:] > self.t] = 1
        loss = self.l1_loss(weight_mask, map_gt)
        return loss


class MaskLoss1(nn.Module):
    def __init__(self, t = 0.1):
        super(MaskLoss1, self).__init__()
        self.t = t

    def convert_L(self, img):
        img_L = (0.114 * img[0, 0, :, :] + 0.587 * img[0, 1, :, :] + \
                 0.299 * img[0, 2, :, :]).unsqueeze(0).unsqueeze(0)
        return img_L

    def l1_loss(self, input,output):
        return torch.mean(torch.abs(input - output))

    def forward(self, weight_mask, map):

        mask_removal, mask_inpainting = torch.split(weight_mask, 3, dim=1)

        map_L = self.convert_L(map)

        map_removal = torch.zeros(map_L.shape).to(map.device)
        map_removal[0, 0, map_L[0,0,:,:] < self.t] = 1
        loss1 = self.l1_loss(mask_removal, map_removal)

        map_inpainting = torch.zeros(map_L.shape).to(map.device)
        map_inpainting[0, 0, map_L[0, 0, :, :] > self.t] = 1
        loss2 = self.l1_loss(mask_inpainting, map_inpainting)

        return loss1 + loss2


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=[0.25,0.75], gamma=2, ignore_index=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.clamp(output, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss() # absorb sigmoid into BCELoss

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        # pdb.set_trace()
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                target_tensor = self.get_target_tensor(input_i, target_is_real)
                loss += self.loss(input_i, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)


class DiscLoss():
    def name(self):
        return 'SGAN'

    def initialize(self, opt, tensor):
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, realA=None, fakeB=None, realB=None):
        pred_fake = None
        pred_real = None
        loss_D_fake = 0
        loss_D_real = 0
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero

        if fakeB is not None:
            pred_fake = net.forward(fakeB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, 0)

        # Real
        if realB is not None:
            pred_real = net.forward(realB)
            loss_D_real = self.criterionGAN(pred_real, 1)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, pred_fake, pred_real


class DiscLossRa(DiscLoss):
    def initialize(self, opt, tensor):
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)

        loss_G = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 0)
        loss_G += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 1)
        return loss_G * 0.5

    def get_g_feat_loss(self, net, realA, fakeB, realB):
        pred_real, feat_real = net.forward(realB, require_feat=True)
        pred_fake, feat_fake = net.forward(fakeB, require_feat=True)

        loss_G = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 0)
        loss_G += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 1)
        loss_Feat = F.l1_loss(feat_fake,feat_real)
        return loss_G * 0.5, loss_Feat

    def get_loss(self, net, realA, fakeB, realB):
        # pdb.set_trace()
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())
        
        loss_D = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 1)
        loss_D += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 0)
        return loss_D * 0.5, pred_fake, pred_real
