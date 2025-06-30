import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

def REC_loss(pred_y, true_y):
    res = (pred_y - true_y) ** 2
    dim = true_y.shape[1]
    mask = torch.sum(torch.abs(true_y), dim=1) > 0
    loss = 1.0 / (torch.sum(mask) * dim) * torch.sum(torch.sum(res, dim=1) * mask)
    return loss

class PolarRegularization(nn.Module):
    def __init__(self, reduction, temperature=1):
        super(PolarRegularization, self).__init__()
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, linear_prob, y):
        prob = torch.softmax(linear_prob, dim=1)
        prob_class0 = prob[:, 0]
        prob_class1 = prob[:, 1]
        diff = (2 * y - 1) * (prob_class1 - prob_class0) #- 0.1 * y
        loss_ = torch.exp((diff - 1) ** 2 / self.temperature)
        # loss_ = (diff - 1) ** 2
        if self.reduction == "mean":
            loss = loss_.sum() / len(y)
        else:
            loss = loss_.sum()

        return loss

class MarginDiff(nn.Module):
    def __init__(self, reduction, tau=1):
        super(MarginDiff, self).__init__()
        self.reduction = reduction
        self.tau = tau

    def forward(self, projection, y):
        if y.is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # projection = projection / torch.norm(projection.detach(), p=2, dim=1, keepdim=True)
        # projection = projection / torch.norm(projection, p=2, dim=1, keepdim=True)
        sim = torch.mm(projection, projection.T)
        # exp_sim = torch.exp(sim)

        same_mask = torch.eq(y.view(-1, 1), y.view(-1, 1).T).float() - torch.eye(len(y)).to(device)
        diff_mask = 1 - torch.eq(y.view(-1, 1), y.view(-1, 1).T).float()


        n = sim.shape[0]
        loss_ = []
        for i in range(n):
            if torch.sum(same_mask[i, :]) != 0:
                sim_same_class_ = sim[i, same_mask[i, :] == 1].min()
            else:
                sim_same_class_ = torch.tensor([0]).to(device)
            if torch.sum(diff_mask[i, :]) != 0:
                sim_diff_class_ = sim[i, diff_mask[i, :] == 1].max()
            else:
                sim_diff_class_ = torch.tensor([0]).to(device)

            loss_i_ = torch.maximum(torch.tensor([0]).to(device), sim_same_class_ +sim_diff_class_ + self.tau)
            loss_.append(loss_i_)
        loss__ = torch.stack(loss_)

        if self.reduction == "mean":
            loss = loss__.mean()
        else:
            loss = loss__.sum()

        return loss


class AdaptiveResidual(nn.Module):
    def __init__(self, beta, C):
        super(AdaptiveResidual, self).__init__()
        self.beta = beta
        self.C = C
        # self.margin_loss = MarginDiff(reduction="sum")

    def forward(self, x, y):
        # x = x / torch.norm(x, dim=1, keepdim=True)
        # y = y / (torch.norm(y, dim=1, keepdim=True) + 1e-5)
        residual = x - y

        orth_loss = torch.norm(torch.mm(x, residual.T))
        mse_loss = torch.sum((torch.norm(residual, dim=1) - self.C)**2)
        # margin_loss = self.margin_loss(residual, label)
        n = x.shape[0]

        loss = 1 / n * (self.beta / 2 * mse_loss + orth_loss)

        return loss




# modify from https://github.com/easezyc/deep-transfer-learning
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMD_loss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def guassian_kernel(self, X, Y):
        """计算Gram核矩阵
        X: sample_size_1 * feature_size 的数据
        Y: sample_size_2 * feature_size 的数据
        kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
        kernel_num: 表示的是多核的数量
        fix_sigma: 表示是否使用固定的标准差
            return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                            矩阵，表达形式:
                            [   K_ss K_st
                                K_ts K_tt ]
        """
        kernel_mul = self.kernel_mul
        kernel_num = self.kernel_num
        fix_sigma = self.fix_sigma

        n_samples = int(X.size()[0]) + int(Y.size()[0])
        total = torch.cat([X, Y], dim=0)  # 合并在一起

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  # 将多个核合并在一起

    def forward(self, X, Y):
        n = int(X.size()[0])
        m = int(Y.size()[0])
        kernels = self.guassian_kernel(X, Y)
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，X<->X
        XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，X<->Y

        YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Y<->X
        YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Y<->Y

        loss = (XX + XY).sum() + (YX + YY).sum()
        return loss


# modify from https://github.com/easezyc/deep-transfer-learning
class HSICLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(HSICLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def guassian_kernel(self, X, Y):
        """计算Gram核矩阵
        X: sample_size_1 * feature_size 的数据
        Y: sample_size_2 * feature_size 的数据
        kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
        kernel_num: 表示的是多核的数量
        fix_sigma: 表示是否使用固定的标准差
            return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                            矩阵，表达形式:
                            [   K_ss K_st
                                K_ts K_tt ]
        """
        kernel_mul = self.kernel_mul
        kernel_num = self.kernel_num
        fix_sigma = self.fix_sigma

        n_samples = int(X.size()[0]) + int(Y.size()[0])
        total = torch.cat([X, Y], dim=0)  # 合并在一起

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  # 将多个核合并在一起

    def forward(self, W, G):
        device = W.device
        n = int(W.size()[0])
        m = int(G.size()[0])
        kernels = self.guassian_kernel(W, G)
        KW = kernels[:n, :n]
        KG = kernels[n:, n:]

        all_one_vec = torch.ones((n, 1))
        H = torch.eye(n, n) - torch.mm(all_one_vec, all_one_vec.T) / n
        H = H.to(device)

        CKW = torch.mm(KW, H)
        CKG = torch.mm(KG, H)

        loss = -torch.trace(torch.mm(CKW, CKG)) / ((n-1) ** 2)

        return loss


def sysmatrix_sqrt(X):
    U, S, V = torch.svd(X)
    sqrtX = torch.mm(torch.mm(U, torch.diag(torch.sqrt(S))), V.t())
    return sqrtX


def tensor_cov(X):
    frac = 1.0 / (X.size(1) - 1)
    mX = X - torch.mean(X, dim=1, keepdim=True)
    return frac * torch.mm(mX, mX.t())


def gauss_wasserstein_dist(X, Y):
    mX = X.mean(dim=0)
    mY = Y.mean(dim=0)
    covX = tensor_cov(X.t())
    covY = tensor_cov(Y.t())
    wp1 = torch.norm(mX - mY) ** 2
    sqrt_covY = sysmatrix_sqrt(covY)
    wp2 = torch.trace(covX + covX - 2 * sysmatrix_sqrt(torch.mm(torch.mm(sqrt_covY, covX), sqrt_covY)))
    gwasser_dist = wp1 + wp2
    return gwasser_dist


def gauss_wasserstein_dist2(X, Y):
    mX = X.mean(dim=0)
    mY = Y.mean(dim=0)
    covX = tensor_cov(X.t())
    covY = tensor_cov(Y.t())
    wp1 = torch.norm(mX - mY) ** 2
    sqrt_covY = sysmatrix_sqrt(covY)
    sqrt_covX = sysmatrix_sqrt(covX)
    wp2 = torch.norm(sqrt_covX - sqrt_covY) ** 2
    gwasser_dist = wp1 + wp2
    return gwasser_dist


class GWassersteinDist(nn.Module):
    def __init__(self):
        super(GWassersteinDist, self).__init__()

    def forward(self, W, G):
        mW = W.mean(dim=0)
        mG = G.mean(dim=0)
        covW = tensor_cov(W.t())
        covG = tensor_cov(G.t())
        wp1 = torch.norm(mW - mG) ** 2
        sqrt_covG = sysmatrix_sqrt(covG)
        wp2 = torch.trace(covW + covG - 2 * sysmatrix_sqrt(torch.mm(torch.mm(sqrt_covG, covW), sqrt_covG)))
        loss = wp1 + wp2
        return loss


class GWassersteinDist2(nn.Module):
    def __init__(self):
        super(GWassersteinDist2, self).__init__()

    def forward(self, W, G):
        mW = W.mean(dim=0)
        mG = G.mean(dim=0)
        covW = tensor_cov(W.t())
        covG = tensor_cov(G.t())
        wp1 = torch.norm(mW - mG) ** 2
        sqrt_covW = sysmatrix_sqrt(covW)
        sqrt_covG = sysmatrix_sqrt(covG)
        wp2 = torch.norm(sqrt_covW - sqrt_covG) ** 2
        loss = wp1 + wp2
        return loss


class SimConsistentLoss2(nn.Module):
    def __init__(self, temperature=0.7):
        super(SimConsistentLoss2, self).__init__()
        self.temperature = temperature

    def forward(self, W, G):
        W = W / torch.norm(W, 2, 1, keepdim=True)
        G = G / (torch.norm(G, 2, 1, keepdim=True) + 1e-6)

        simW = torch.exp(torch.mm(W, W.T) / self.temperature)
        simG = torch.exp(torch.mm(G, G.T) / self.temperature)

        norm_simW = simW / torch.sum(simW, dim=1, keepdim=True)
        norm_simG = simG / torch.sum(simG, dim=1, keepdim=True)

        loss = torch.norm(norm_simW - norm_simG)

        # logits_norm_simW = torch.log(norm_simW)
        #
        # kl_div = torch.nn.KLDivLoss(reduction="sum")
        #
        # loss =  kl_div(logits_norm_simW, norm_simG)

        return loss


class SuperviseContrastive(nn.Module):
    def __init__(self, temperature=0.7):
        super(SuperviseContrastive, self).__init__()
        self.temperature = temperature

    def forward(self, projection, y):
        if y.is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if len(y.unique()) == 1:
            loss = torch.tensor([0]).to(device)
        else:
            projection = projection / torch.norm(projection, p=2, dim=1, keepdim=True)
            sim = torch.div(torch.mm(projection, projection.T), self.temperature)
            exp_sim = torch.exp(sim) - torch.diag(torch.diag(sim.detach()))
            same_mask = torch.eq(y.view(-1, 1), y.view(-1, 1).T).float() - torch.eye(len(y)).to(device)
            exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)
            log_sim_mask = sim * same_mask

            log_sum_exp_sim = torch.log(exp_sim_sum)

            div_log_sum = log_sim_mask - log_sum_exp_sim
            div_sim = torch.sum(div_log_sum, dim=1, keepdim=True)
            div_sim_ = -div_sim / torch.maximum(same_mask.sum(dim=1, keepdim=True), torch.tensor([1]).to(device))
            loss = div_sim_.mean()

        return loss


class HSICLoss2(nn.Module):
    def __init__(self, temperature=0.7):
        super( HSICLoss2, self).__init__()
        self.temperature = temperature

    def forward(self, W, G):
        W = W / torch.norm(W, 2, 1, keepdim=True)
        G = G / (torch.norm(G, 2, 1, keepdim=True) + 1e-6)

        KW = torch.exp(torch.mm(W, W.T) / self.temperature)
        KG = torch.exp(torch.mm(G, G.T) / self.temperature)

        device = W.device
        n = int(W.size()[0])

        all_one_vec = torch.ones((n, 1))
        H = torch.eye(n, n) - torch.mm(all_one_vec, all_one_vec.T) / n
        H = H.to(device)

        CKW = torch.mm(KW, H)
        CKG = torch.mm(KG, H)

        loss = torch.trace(torch.mm(CKW, CKG)) / ((n-1) ** 2)

        return loss


class SimConsistentLoss(nn.Module):
    def __init__(self, temperature=0.7):
        super(SimConsistentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, W, G):
        # W = W / torch.norm(W, 2, 1, keepdim=True)
        # G = G / (torch.norm(G, 2, 1, keepdim=True) + 1e-6)

        simW = torch.exp(torch.mm(W, W.T) / self.temperature)
        simG = torch.exp(torch.mm(G, G.T) / self.temperature)

        norm_simW = simW / torch.sum(simW, dim=1, keepdim=True)
        norm_simG = simG / torch.sum(simG, dim=1, keepdim=True)

        logits_norm_simW = torch.log(norm_simW)

        kl_div = torch.nn.KLDivLoss(reduction="sum")

        loss = kl_div(logits_norm_simW, norm_simG)

        return loss


class PolarSimReg(nn.Module):
    def __init__(self, reduction, temperature=0.7):
        super(PolarSimReg, self).__init__()
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, projection, y):
        if y.is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        projection = projection / torch.norm(projection, p=2, dim=1, keepdim=True)
        # projection = projection / torch.norm(projection, p=2, dim=1, keepdim=True)
        sim = torch.mm(projection, projection.T)
        # exp_sim = torch.exp(sim)

        same_mask = torch.eq(y.view(-1, 1), y.view(-1, 1).T).float() - torch.eye(len(y)).to(device)
        diff_mask = 1 - torch.eq(y.view(-1, 1), y.view(-1, 1).T).float()

        n = sim.shape[0]
        loss_ = []
        for i in range(n):
            if torch.sum(same_mask[i, :]) != 0:
                sim_same_class_ = sim[i, same_mask[i, :] == 1]
            else:
                sim_same_class_ = torch.tensor([0]).to(device)
            if torch.sum(diff_mask[i, :]) != 0:
                sim_diff_class_ = sim[i, diff_mask[i, :] == 1]
            else:
                sim_diff_class_ = torch.tensor([0]).to(device)

            div = sim_diff_class_.view(1, -1) - sim_same_class_.view(1, -1).T
            exp_div = torch.exp((div + 0.5) / self.temperature)
            loss_i_ = torch.log(1 + exp_div.sum())
            loss_.append(loss_i_)
        loss__ = torch.stack(loss_)

        if self.reduction == "mean":
            loss = loss__.mean()
        else:
            loss = loss__.sum()

        return loss

class PolarRegularization(nn.Module):
    def __init__(self, reduction, temperature=1):
        super(PolarRegularization, self).__init__()
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, linear_prob, y):
        prob = torch.softmax(linear_prob, dim=1)
        prob_class0 = prob[:, 0]
        prob_class1 = prob[:, 1]
        diff = (2 * y - 1) * (prob_class1 - prob_class0) #- 0.1 * y
        loss_ = torch.exp((diff - 1) ** 2 / self.temperature)
        # loss_ = (diff - 1) ** 2
        if self.reduction == "mean":
            loss = loss_.sum() / len(y)
        else:
            loss = loss_.sum()

        return loss



