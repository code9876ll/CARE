import torch
import torch.nn as nn
import random

def calc_style_loss(self, input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    mse_loss = nn.MSELoss()
    return mse_loss(input_mean, target_mean) + \
           mse_loss(input_std, target_std)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    # print('len(size):',len(size))
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    # feat_var = feat.view(N, C, -1).var(dim=1) + eps
    # print('feat_var:',feat_var,feat_var.shape)
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    # print('feat_std:', feat_std, feat_std.shape)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    # print('feat_mean:', feat_mean, feat_mean.shape)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    # print('content_feat.size()[:2], style_feat.size()[:2]:',content_feat.size()[:2], style_feat.size()[:2])
    if (content_feat.size()[:2] != style_feat.size()[:2]):
        B_c, _ = content_feat.size()[:2]
        B_s, _ = style_feat.size()[:2]
        # print('B_c,B_s',B_c,B_s)
        index = torch.LongTensor(random.sample(range(B_s), B_c)).to("cuda")
        # print('style_img:',style_feat[0])
        # print('index:', index)
        style_feat = torch.index_select(style_feat, 0, index)
    # assert (content_feat.size()[:2] == style_feat.size()[:2])
    # print('content_feat, style_feat:', content_feat[0], style_feat[0])
    size = content_feat.size()
    # print('size:',size)#16 768
    style_mean, style_std = calc_mean_std(style_feat)
    # print('style_mean, style_std',style_mean[0], style_std[0])
    # print('style_mean, style_std', style_mean.shape, style_std.shape)
    content_mean, content_std = calc_mean_std(content_feat)
    # print('content_mean, content_std', content_mean[0], content_std[0])
    # print('content_mean, content_std', content_mean.shape, content_std.shape)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    # print('normalized_feat',normalized_feat[0])
    # gg = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    # print('gg',gg[0],gg.shape)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# def calc_mean_std(feat, eps=1e-5):
#     # eps is a small value added to the variance to avoid divide-by-zero.
#     size = feat.size()
#     print('len(size):',len(size))
#     # assert (len(size) == 4)
#     N, C = size[:2]
#     # feat_var = feat.view(N, C, -1).var(dim=2) + eps
#     feat_var = feat.view(N, C, -1).var(dim=1) + eps
#     # print('feat_var:',feat_var,feat_var.shape)
#     feat_std = feat_var.sqrt()#.view(N, C, 1, 1)
#     # print('feat_std:', feat_std, feat_std.shape)
#     feat_mean = feat.view(N, C, -1).mean(dim=1)#.view(N, C, 1, 1)
#     # print('feat_mean:', feat_mean, feat_mean.shape)
#     return feat_mean, feat_std
#
#
# def adaptive_instance_normalization(content_feat, style_feat):
#     # print('content_feat.size()[:2], style_feat.size()[:2]:',content_feat.size()[:2], style_feat.size()[:2])
#     # assert (content_feat.size()[:2] == style_feat.size()[:2])
#     print('content_feat, style_feat:', content_feat[0], style_feat[0])
#     size = content_feat.size()
#     # print('size:',size)#16 768
#     style_mean, style_std = calc_mean_std(style_feat)
#     print('style_mean, style_std',style_mean, style_std)
#     content_mean, content_std = calc_mean_std(content_feat)
#     print('content_mean, content_std', content_mean, content_std)
#
#     normalized_feat = (content_feat - content_mean.expand(
#         size)) / content_std.expand(size)
#     print('normalized_feat',normalized_feat)
#     print('gg',normalized_feat * style_std.expand(size) + style_mean.expand(size))
#     return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())
