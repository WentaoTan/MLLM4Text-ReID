import torch
from torch import nn
from torch.nn import functional as F

from torch.distributions import Normal, Uniform

import random
import numpy as np


class NoiseStyle(nn.Module):
    def __init__(self, p=1.0, alpha=0.1, eps=1e-6, mix='random'):

        super().__init__()
        self.p = p
        self.noise = torch.distributions.Normal(0., 0.1)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'NoiseStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        mu_noise = self.noise.sample((B, x.size(1), 1, 1)).cuda()
        sig_noise = self.noise.sample((B, x.size(1), 1, 1)).cuda()

        mu_mix = mu + mu_noise
        sig_mix = sig + sig_noise
        return x_normed * sig_mix + mu_mix


class MixStyle(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        # self.beta = torch.distributions.uniform.Uniform(low=0.,high=1.)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)
        # lmda = random.random()

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        elif self.mix == 'crossID':
            perm = torch.arange(B)
            perm_ID = perm.chunk(B // 4, dim=0)
            for i in range(len(perm_ID)):
                random_index = torch.randperm(perm_ID[i].size(0))
                perm_ID[i].data.copy_(perm_ID[i][random_index])
            perm = torch.cat(perm_ID)
        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix


class AdaIN(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        # self.beta = torch.distributions.uniform.Uniform(low=0.,high=1.)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'AdaIN(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]

        return x_normed * sig2 + mu2


class DirBatchStyle(nn.Module):

    def __init__(self, p=1.0, alpha=0.1, eps=1e-6, type='dir_bat'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.dirichlet = torch.distributions.Dirichlet(torch.tensor([0.1, 0.1, 0.1]))
        self.eps = eps
        self.alpha = alpha
        self.type = type


    def __repr__(self):
        return f'DirBatchStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, type={self.type})'

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        if isinstance(x, list):
            style = x[1]
            x = x[0]

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        mu_1 = x.mean(dim=[2, 3], keepdim=True)
        std_1 = x.std(dim=[2, 3], keepdim=True)
        # mu_1 = torch.cat([x,style]).mean(dim=[2, 3], keepdim=True)
        # std_1 = torch.cat([x,style]).std(dim=[2, 3], keepdim=True)
        mu_mu = mu_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        mu_std = mu_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        std_mu = std_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        std_std = std_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        # Distri_mu = Normal(mu_mu, F.softplus(mu_std))
        # Distri_std = Normal(std_mu, F.softplus(std_std))
        Distri_mu = Normal(mu_mu, mu_std)
        Distri_std = Normal(std_mu, std_std)

        mu_b = Distri_mu.sample([B, ])
        sig_b = Distri_std.sample([B, ])
        mu_b = mu_b.unsqueeze(2).unsqueeze(2)
        sig_b = sig_b.unsqueeze(2).unsqueeze(2)
        mu_b, sig_b = mu_b.detach(), sig_b.detach()

        perm1 = torch.randperm(B)
        perm2 = torch.randperm(B)
        mu2, sig2 = mu[perm1], sig[perm1]
        mu3, sig3 = mu[perm2], sig[perm2]

        lmda = self.dirichlet.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)
        mu_mix = mu * lmda[:, :, :, :, 0] + mu2 * lmda[:, :, :, :, 1] + mu3 * lmda[:, :, :, :, 2]
        sig_mix = sig * lmda[:, :, :, :, 0] + sig2 * lmda[:, :, :, :, 1] + sig3 * lmda[:, :, :, :, 2]

        # a = random.uniform(0,1)
        # b = random.uniform(0,1-a)
        # c = 1-a-b
        # weight_list = [a,b,c]
        # random.shuffle(weight_list)
        # mu_mix = mu * weight_list[0] + mu2 * weight_list[1] + mu3 * weight_list[2]
        # sig_mix = sig * weight_list[0] + sig2 * weight_list[1] + sig3 * weight_list[2]

        # mu_mix = mu * 0.33 + mu2 * 0.33 + mu3 * 0.34
        # sig_mix = sig * 0.33 + sig2 * 0.33 + sig3 * 0.34

        # lmda = self.dirichlet.sample((B, 1, 1, 1))
        # lmda = lmda.to(x.device)
        # lmda = lmda.squeeze()
        # _, index = torch.sort(lmda)
        # lmda = lmda.gather(-1, index)
        # lmda = lmda.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        #
        # mu_mix = mu * lmda[:, :, :, :, 2] + mu2 * lmda[:, :, :, :, 1] + mu3 * lmda[:, :, :, :, 0]
        # sig_mix = sig * lmda[:, :, :, :, 2] + sig2 * lmda[:, :, :, :, 1] + sig3 * lmda[:, :, :, :, 0]

        if self.type == 'dir':
            return x_normed * sig_mix + mu_mix
        elif self.type == 'bat':
            return x_normed * sig_b + mu_b
        elif self.type == 'dir_bat':
            z = 0.9
            sig_mix_f = z * sig_mix + (1 - z) * sig_b
            mu_mix_f = z * mu_mix + (1 - z) * mu_b

            return x_normed * sig_mix_f + mu_mix_f

        else:
            raise print("type error")

class UBS(nn.Module):

    def __init__(self, p=1.0, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.eps = eps


    def __repr__(self):
        return f'UBS(p={self.p})'

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        mu_1 = x.mean(dim=[2, 3], keepdim=True)
        std_1 = x.std(dim=[2, 3], keepdim=True)
        # mu_1 = torch.cat([x,style]).mean(dim=[2, 3], keepdim=True)
        # std_1 = torch.cat([x,style]).std(dim=[2, 3], keepdim=True)
        mu_mu = mu_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        mu_std = mu_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        std_mu = std_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        std_std = std_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        mu_std.data.clamp_(min=self.eps)
        std_std.data.clamp_(min=self.eps)


        Distri_mu = Uniform(mu_mu-3*mu_std, mu_mu+3*mu_std)
        Distri_std = Uniform(std_mu-3*std_std, std_mu+3*std_std)

        mu_b = Distri_mu.sample([B, ])
        sig_b = Distri_std.sample([B, ])
        mu_b = mu_b.unsqueeze(2).unsqueeze(2)
        sig_b = sig_b.unsqueeze(2).unsqueeze(2)
        mu_b, sig_b = mu_b.detach(), sig_b.detach()

        return x_normed * sig_b + mu_b

class MemoryDBS(nn.Module):

    def __init__(self, p=1.0, alpha=0.1, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.dirichlet = torch.distributions.Dirichlet(torch.tensor([0.1, 0.1, 0.1]))
        self.eps = eps
        self.alpha = alpha
        self.z = 0.5
        self.register_buffer('mu_mu', torch.zeros([256]))
        self.register_buffer('mu_std', torch.ones([256]))
        self.register_buffer('std_mu', torch.zeros([256]))
        self.register_buffer('std_std', torch.ones([256]))

    def __repr__(self):
        return f'MemoryDBS(p={self.p}, alpha={self.alpha}, eps={self.eps},z={self.z})'

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        mu_1 = x.mean(dim=[2, 3], keepdim=True)
        std_1 = x.std(dim=[2, 3], keepdim=True)
        # mu_1 = torch.cat([x,style]).mean(dim=[2, 3], keepdim=True)
        # std_1 = torch.cat([x,style]).std(dim=[2, 3], keepdim=True)
        mu_mu = mu_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        mu_std = mu_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        std_mu = std_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        std_std = std_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        self.mu_mu.data.mul_(0.9).add_(1 - 0.9, mu_mu.data)
        self.mu_std.data.mul_(0.9).add_(1 - 0.9, F.softplus(mu_std.data))
        self.std_mu.data.mul_(0.9).add_(1 - 0.9, std_mu.data)
        self.std_std.data.mul_(0.9).add_(1 - 0.9, F.softplus(std_std.data))
        Distri_mu = Normal(self.mu_mu, self.mu_std)
        Distri_std = Normal(self.std_mu, self.std_std)
        mu_b = Distri_mu.sample([B, ])
        sig_b = Distri_std.sample([B, ])
        mu_b = mu_b.unsqueeze(2).unsqueeze(2)
        sig_b = sig_b.unsqueeze(2).unsqueeze(2)
        mu_b, sig_b = mu_b.detach(), sig_b.detach()

        perm1 = torch.randperm(B)
        perm2 = torch.randperm(B)
        mu2, sig2 = mu[perm1], sig[perm1]
        mu3, sig3 = mu[perm2], sig[perm2]

        lmda = self.dirichlet.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)
        mu_mix = mu * lmda[:, :, :, :, 0] + mu2 * lmda[:, :, :, :, 1] + mu3 * lmda[:, :, :, :, 2]
        sig_mix = sig * lmda[:, :, :, :, 0] + sig2 * lmda[:, :, :, :, 1] + sig3 * lmda[:, :, :, :, 2]

        sig_mix_f = self.z * sig_mix + (1 - self.z) * sig_b
        mu_mix_f = self.z * mu_mix + (1 - self.z) * mu_b

        return x_normed * sig_mix_f + mu_mix_f


# class DSU(nn.Module):
#
#     def __init__(self, p=1.0, eps=1e-6):
#         """
#         Args:
#           p (float): probability of using MixStyle.
#           alpha (float): parameter of the Beta distribution.
#           eps (float): scaling parameter to avoid numerical issues.
#           mix (str): how to mix.
#         """
#         super().__init__()
#         self.p = p
#         self.eps = eps
#
#     def __repr__(self):
#         return f'DSU(p={self.p}, eps={self.eps})'
#
#     def forward(self, x):
#         if not self.training:
#             return x
#
#         if random.random() > self.p:
#             return x
#
#         mu = x.mean(dim=[2, 3], keepdim=True)
#         var = x.var(dim=[2, 3], keepdim=True)
#         sig = (var + self.eps).sqrt()
#         mu, sig = mu.detach(), sig.detach()
#         x_normed = (x - mu) / sig
#
#         C = x.size(1)
#
#         mu_std = mu.std(dim=0).squeeze()
#         std_std = sig.std(dim=0).squeeze()
#         noise_mu = Normal(0, 1).sample([C, ]).cuda()
#         noise_sig = Normal(0, 1).sample([C, ]).cuda()
#         dsu_mu = (noise_mu * mu_std)[None, :, None, None] + mu
#         dsu_sig = (noise_sig * std_std)[None, :, None, None] + sig
#
#         return x_normed * dsu_sig + dsu_mu
class DSU(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DSU, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 4.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

class ABS(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.beta = torch.distributions.Beta(alpha, alpha)
        # self.beta = torch.distributions.uniform.Uniform(low=0.,high=1.)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'AdaIN(alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if isinstance(x, list):
            style = x[1]
            x = x[0]
            mu_style = style.mean(dim=[2, 3], keepdim=True)
            var_style = style.var(dim=[2, 3], keepdim=True)
            sig_style = (var_style + self.eps).sqrt()
            mu_style, sig_style = mu_style.detach(), sig_style.detach()

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]

        mu_1 = mu.clone().detach()
        var_1 = var.clone().detach()
        mu_mu = mu_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        mu_std = mu_1.std(dim=0, keepdim=True, unbiased=True).squeeze(0).squeeze(1).squeeze(1)

        var_mu = var_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        var_std = var_1.std(dim=0, keepdim=True, unbiased=True).squeeze(0).squeeze(1).squeeze(1)

        Distri_mu = Normal(mu_mu, mu_std)
        Distri_var = Normal(var_mu, var_std)

        mu_b = Distri_mu.sample([B, ])
        sig_b = Distri_var.sample([B, ])
        mu_b = mu_b.unsqueeze(2).unsqueeze(2)
        sig_b = sig_b.unsqueeze(2).unsqueeze(2)
        mu_b, sig_b = mu_b.detach(), sig_b.detach()

        z = 0.9
        sig_mix_f = z * sig2 + (1 - z) * sig_b
        mu_mix_f = z * mu2 + (1 - z) * mu_b
        return x_normed * sig_mix_f + mu_mix_f
class EFDM(nn.Module):
    def __init__(self, p=1.0, alpha=0.1, eps=1e-6, mix='random'):

        super().__init__()
        self.p = p
        self.noise = torch.distributions.Normal(0., 0.1)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'EFDM(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def exact_feature_distribution_matching(self, content, style):
        assert (content.size() == style.size())
        B, C, W, H = content.size()
        _, index_content = torch.sort(content.view(B, C, -1))
        value_style, _ = torch.sort(style.view(B, C, -1))
        inverse_index = index_content.argsort(-1)
        transferred_content = content.view(B, C, -1) + \
                              (value_style.gather(-1, inverse_index) - content.view(B, C,
                                                                                    -1)).detach()  # there is no gradient in IL so .detach() can be omitted
        return transferred_content.view(B, C, W, H)

    def exact_feature_distribution_matching_mix(self, content, style):
        assert (content.size() == style.size())
        B, C, W, H = content.size()

        beta = torch.distributions.Beta(0.1, 0.1)
        lmda = beta.sample((B, 1, 1))
        lmda = lmda.to(content.device)
        _, index_content = torch.sort(content.view(B, C, -1))
        value_style, _ = torch.sort(style.view(B, C, -1))
        inverse_index = index_content.argsort(-1)

        transferred_content = content.view(B, C, -1) + (1 - lmda) * (
                value_style.gather(-1, inverse_index) - content.view(B, C,
                                                                     -1)).detach()  # there is no gradient in IL so .detach() can be omitted
        return transferred_content.view(B, C, W, H)

    def exact_feature_distribution_matching_batch(self, content):
        B, C, W, H = content.size()
        _, index_content = torch.sort(content.view(B, C, -1))
        histogram_batch = content.permute(1, 0, 2, 3).reshape(C, -1)
        histogram_sample = []
        for i in range(B):
            sample_index = torch.cat([torch.randperm(B * W * H)[:W * H].unsqueeze(0) for i in range(C)], dim=0).cuda()
            histogram_sample.append(histogram_batch.gather(-1, sample_index).unsqueeze(0))
        histogram_sample = torch.cat(histogram_sample, dim=0)
        value_style, _ = torch.sort(histogram_sample)
        inverse_index = index_content.argsort(-1)
        transferred_content = content.view(B, C, -1) + (value_style.gather(-1, inverse_index) - content.view(B, C,
                                                                                                             -1)).detach()
        return transferred_content.view(B, C, W, H)

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x
        content = x
        style = x.detach()
        assert (content.size() == style.size())
        B, C, W, H = content.size()
        perm = torch.randperm(B)
        style = style[perm]
        Noise_weight = Normal(1., 0.1)
        Noise_weight = Noise_weight.sample([B, C, W, H]).squeeze().cuda()
        style = Noise_weight * style
        _, index_content = torch.sort(content.view(B, C, -1))
        value_style, _ = torch.sort(style.view(B, C, -1))
        inverse_index = index_content.argsort(-1)
        transferred_content = content.view(B, C, -1) + \
                              (value_style.gather(-1, inverse_index) - content.view(B, C,
                                                                                    -1)).detach()  # there is no gradient in IL so .detach() can be omitted
        return transferred_content.view(B, C, W, H)


class EFDMix(nn.Module):
    def __init__(self, p=1.0, alpha=0.1, eps=1e-6, mix='random'):

        super().__init__()
        self.p = p

        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.beta = torch.distributions.Beta(0.1, 0.1)

    def __repr__(self):
        return f'EFDMix(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x
        content = x
        style = x.detach()
        assert (content.size() == style.size())
        B, C, W, H = content.size()
        perm = torch.randperm(B)
        style = style[perm]
        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(content.device)
        _, index_content = torch.sort(content.view(B, C, -1))
        value_style, _ = torch.sort(style.view(B, C, -1))
        inverse_index = index_content.argsort(-1)
        transferred_content = content.view(B, C, -1) + \
                              (1 - lmda) * (value_style.gather(-1, inverse_index) - content.view(B, C, -1)).detach()
        return transferred_content.view(B, C, W, H)


class EFDMBatch(nn.Module):
    def __init__(self, p=1.0, alpha=0.1, eps=1e-6, mix='random'):

        super().__init__()
        self.p = p
        self.noise = torch.distributions.Normal(0., 0.1)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.register_buffer('batch_value', torch.rand([48, 256, 2048]))

    def __repr__(self):
        return f'EFDMBatch(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x
        content = x
        style = x.detach()
        assert (content.size() == style.size())
        B, C, W, H = content.size()
        perm = torch.randperm(B)
        style = style[perm]
        _, index_content = torch.sort(content.view(B, C, -1))
        value_style, _ = torch.sort(style.view(B, C, -1))
        self.batch_value.data.copy_(0.9 * self.batch_value + 0.1 * value_style)
        inverse_index = index_content.argsort(-1)
        transferred_content = content.view(B, C, -1) + \
                              (self.batch_value.gather(-1, inverse_index) - content.view(B, C,
                                                                                         -1)).detach()  # there is no gradient in IL so .detach() can be omitted
        return transferred_content.view(B, C, W, H)


class ValueStyle(nn.Module):

    def __init__(self, p=1.0, alpha=0.1, eps=1e-6, type='noise'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.eps = eps
        self.alpha = alpha
        self.type = type

    def __repr__(self):
        return f'ValueStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, type={self.type})'

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B, C, W, H = x.size()
        _, index_content = torch.sort(x.view(B, C, -1))
        inverse_index = index_content.argsort(-1)

        if self.type == 'noise':
            # zero = torch.tensor([0.]).to(x.device)
            # one = torch.tensor([0.1]).to(x.device)
            zero = x.mean()
            one = x.std()

            noise_style = 0.1 * Normal(zero, one).sample([B, C, W, H]).squeeze() + 0.9 * x.detach()
            value_style, _ = torch.sort(noise_style.view(B, C, -1))

        elif self.type == 'batch':
            # mu = x.mean(dim=[0, 2, 3])
            # std = x.std(dim=[0, 2, 3])
            # Distri = Normal(mu, std)
            # batch_sample_style = Distri.sample([B*W*H,])
            #
            # batch_sample_style = (batch_sample_style.reshape(B,W,H,C).permute(0,3,1,2) + x.detach())/2
            random_shuffle_x = x.view(-1).cpu().detach().numpy()
            np.random.shuffle(random_shuffle_x)
            batch_sample_style = torch.from_numpy(random_shuffle_x).to(x.device)
            value_style, _ = torch.sort(batch_sample_style.view(B, C, -1))
        else:
            raise print("type error")

        transferred_x = x.view(B, C, -1) + (value_style.gather(-1, inverse_index) - x.view(B, C, -1)).detach()
        return transferred_x.view(B, C, W, H)


class MultiStyle(nn.Module):

    def __init__(self, p=1.0, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.eps = eps
        self.alpha = alpha
        self.type = ['AdaIN', 'Mixstyle', 'DBS']

    def __repr__(self):
        return f'MultiStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, type={self.type})'

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        type = random.sample(self.type, 1)[0]
        B, C, W, H = x.size()
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        if type == 'AdaIN':
            perm = torch.randperm(B)
            AdaIN_mu = mu[perm]
            AdaIN_sig = sig[perm]
            return AdaIN_sig * x_normed + AdaIN_mu
        elif type == 'Mixstyle':
            perm = torch.randperm(B)
            AdaIN_mu = mu[perm]
            AdaIN_sig = sig[perm]
            lmda = torch.distributions.Beta(0.1, 0.1).sample((B, 1, 1, 1)).to(x.device)
            mu_mix = mu * lmda + AdaIN_mu * (1 - lmda)
            sig_mix = sig * lmda + AdaIN_sig * (1 - lmda)
            return x_normed * sig_mix + mu_mix
        elif type == 'DBS':
            mu_1 = x.mean(dim=[2, 3], keepdim=True)
            std_1 = x.std(dim=[2, 3], keepdim=True)
            mu_mu = mu_1.mean(dim=0, keepdim=True).squeeze()
            mu_std = mu_1.std(dim=0, keepdim=True).squeeze()
            std_mu = std_1.mean(dim=0, keepdim=True).squeeze()
            std_std = std_1.std(dim=0, keepdim=True).squeeze()
            Distri_mu = Normal(mu_mu, mu_std)
            Distri_var = Normal(std_mu, std_std)
            mu_b = Distri_mu.sample([B, ])
            sig_b = Distri_var.sample([B, ])
            mu_b = mu_b.unsqueeze(2).unsqueeze(2)
            sig_b = sig_b.unsqueeze(2).unsqueeze(2)
            mu_b, sig_b = mu_b.detach(), sig_b.detach()
            return sig_b * x_normed + mu_b
        else:
            raise print("type error")


class visualDBS(nn.Module):

    def __init__(self, p=1.0, alpha=0.1, eps=1e-6, type='dir'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.dirichlet = torch.distributions.Dirichlet(torch.tensor([0.1, 0.1, 0.1]))
        self.eps = eps
        self.alpha = alpha
        self.type = type
        self._activated = True
        self.gap = nn.AdaptiveAvgPool2d(1)

    def __repr__(self):
        return f'visualDBS(p={self.p}, alpha={self.alpha}, eps={self.eps}, type={self.type})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        # if not self.training or not self._activated:
        #     return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()

        mu_1 = x.mean(dim=[2, 3], keepdim=True)
        var_1 = x.std(dim=[2, 3], keepdim=True)
        mu_mu = mu_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        mu_std = mu_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        var_mu = var_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        var_std = var_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        Distri_mu = Normal(mu_mu, torch.full_like(mu_mu, 1.))
        Distri_var = Normal(var_mu, torch.full_like(mu_mu, 1.))
        mu_b = Distri_mu.sample([B, ])
        sig_b = Distri_var.sample([B, ])
        mu_b = mu_b.unsqueeze(2).unsqueeze(2)
        sig_b = sig_b.unsqueeze(2).unsqueeze(2)
        mu_b, sig_b = mu_b.detach(), sig_b.detach()

        mu_center_list = mu.chunk(3, dim=0)
        var_center_list = sig.chunk(3, dim=0)
        mu_center = torch.cat([mu_center_list[0].mean(0, keepdim=True), mu_center_list[1].mean(0, keepdim=True),
                               mu_center_list[2].mean(0, keepdim=True)], dim=0)
        sig_center = torch.cat([var_center_list[0].mean(0, keepdim=True), var_center_list[1].mean(0, keepdim=True),
                                var_center_list[2].mean(0, keepdim=True)], dim=0)
        x_normed = (x - mu) / sig

        # random shuffle
        perm1 = torch.randperm(B)
        perm2 = torch.randperm(B)

        mu2, sig2 = mu[perm1], sig[perm1]
        mu3, sig3 = mu[perm2], sig[perm2]

        lmda = self.dirichlet.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        # mu_mix = mu * 0.33 + mu2 * 0.33 + mu3 * 0.34
        # sig_mix = sig * 0.33 + sig2 * 0.33 + sig3 * 0.34

        mu_mix = mu * lmda[:, :, :, :, 0] + mu2 * lmda[:, :, :, :, 1] + mu3 * lmda[:, :, :, :, 2]
        sig_mix = sig * lmda[:, :, :, :, 0] + sig2 * lmda[:, :, :, :, 1] + sig3 * lmda[:, :, :, :, 2]

        if self.type == 'dir':

            return x_normed * sig_mix + mu_mix

        elif self.type == 'dir_bat':
            z = 0.9
            sig_mix_f = z * sig_mix + (1 - z) * sig_b
            mu_mix_f = z * mu_mix + (1 - z) * mu_b

            return x_normed * sig_mix_f + mu_mix_f
        elif self.type == 'style_inf':
            # sig_mix_list = sig_mix.chunk(3,dim=0)
            # mu_mix_list = mu_mix.chunk(3,dim=0)
            z = 0.9
            sig_mix_f = z * sig_mix + (1 - z) * sig_b
            mu_mix_f = z * mu_mix + (1 - z) * mu_b
            # z = 0.85
            # mu_center = z*mu_center + (1-z) * mu_b.mean(dim=0)[None,:]
            # sig_center = z*sig_center+(1-z)*sig_b.mean(dim=0)[None,:]
            # sig_mix_f = torch.cat([z * sig_mix_list[0] + (1 - z) * sig_b0,z* sig_mix_list[1] + (1 - z) * sig_b1,z * sig_mix_list[2] + (1 - z) * sig_b2],dim=0)
            # mu_mix_f = torch.cat([z * mu_mix_list[0] + (1 - z) * mu_b0,z * mu_mix_list[1] + (1 - z) * mu_b1,z * mu_mix_list[2] + (1 - z) * mu_b2],dim=0)
            import IPython
            IPython.embed()
            return torch.cat([mu, sig], dim=1), torch.cat([mu_mix, sig_mix], dim=1), torch.cat([mu_b, sig_b], dim=1), \
                   torch.cat([mu_mix_f, sig_mix_f], dim=1), self.gap(x), self.gap(x_normed * sig_mix + mu_mix), \
                   self.gap(x_normed * sig_b + mu_b), self.gap(x_normed * sig_mix_f + mu_mix_f), \
                   torch.cat([mu_center, sig_center], dim=1)

        else:
            raise print("type error")

class DiverseStyle(nn.Module):

    def __init__(self, p=0.5,eps=1e-6):
        super().__init__()
        self.p = p
        self.eps = eps
    def __repr__(self):
        return f'DiverseStyle(p={self.p}, eps={self.eps})'

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        # mu = torch.cat([mu,mu_style],dim=0)
        # sig = torch.cat([sig,sig_style],dim=0)

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]

        return x_normed * sig2 + mu2