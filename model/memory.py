import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
import numpy as np


class MC(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        return grad_inputs, None, None, None


def mc(inputs, indexes, features, momentum=0.5):
    return MC.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class MemoryClassifier(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(MemoryClassifier, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features_txt', torch.zeros(num_samples, num_features))
        self.register_buffer('features_img', torch.zeros(num_samples, num_features))
        # self.register_buffer('cam_features',torch.zeros(num_samples,num_features,16,8))

    def MomentumUpdate(self, img, txt, indexes):
        # momentum update
        for im,tx, y in zip(img, txt, indexes):
            self.features_img[y] = self.momentum * self.features_img[y] + (1. - self.momentum) * im
            self.features_img[y] = self.features_img[y] / self.features_img[y].norm()

            self.features_txt[y] = self.momentum * self.features_txt[y] + (1. - self.momentum) * tx
            self.features_txt[y] = self.features_txt[y] / self.features_txt[y].norm()

    def forward(self, img, txt , indexes):
        sim_i2t = mc(img, indexes, self.features_txt, self.momentum)  ## B * C
        sim_i2i = mc(img, indexes, self.features_img, self.momentum)
        sim_t2i = mc(txt, indexes, self.features_img, self.momentum)
        sim_t2t = mc(txt, indexes, self.features_txt, self.momentum)

        loss_i2t = F.cross_entropy(sim_i2t / self.temp, indexes)
        loss_i2i = F.cross_entropy(sim_i2i / self.temp, indexes)
        loss_t2i = F.cross_entropy(sim_t2i / self.temp, indexes)
        loss_t2t = F.cross_entropy(sim_t2t / self.temp, indexes)
        return loss_i2t ,loss_i2i,loss_t2i,loss_t2t


class Memory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp

        # self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('cam_features', torch.zeros(num_samples, num_features, 16, 8))

    def MomentumUpdate(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.cam_features[y] = self.momentum * self.cam_features[y] + (1. - self.momentum) * x
            self.cam_features[y] = self.cam_features[y] / self.cam_features[y].norm()

    def forward(self, inputs, indexes):
        return 0
