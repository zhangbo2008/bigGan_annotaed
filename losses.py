import torch
import torch.nn.functional as F


# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2


def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))  # dis_real表示真实图片输入进去之后, 得到的真实性得分. 这个函数再dis_real 大于等于1的时候 loss都是0. 所以让dis_real 趋近于大于等于1
    loss_fake = torch.mean(F.relu(1. + dis_fake))  # 让dis_fake趋近于小于等于-1.
    return loss_real, loss_fake


# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
# loss = torch.mean(F.relu(1. - dis_real))
# loss += torch.mean(F.relu(1. + dis_fake))
# return loss


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake) #  等价于让dis_fake 越大越好.
    return loss


# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
