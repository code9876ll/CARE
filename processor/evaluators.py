from __future__ import print_function, absolute_import
import torch
import numpy as np
from utils import to_torch

def extract_feature(model, inputs, camids, target_view):
    inputs = to_torch(inputs).cuda()
    # outputs = model(inputs)
    # print('inputs.type():',inputs.type())#torch.cuda.FloatTensor

    # res50
    # outputs = model(inputs)
    outputs = model(inputs, cam_label=camids, view_label=target_view)

    outputs = outputs.data.cpu()
    return outputs

def extract_features(model, data_loader, print_freq=50):
    model.eval()

    features = []
    # global_labels = []
    pids=[]

    with torch.no_grad():
        # for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
        # for i, (imgs, pid, _, fname) in enumerate(data_loader):
        # for i, data in enumerate(data_loader):
        for i, (img, pid, camid, camids, target_view, imgpath) in enumerate(data_loader):
            # print('f:',imgpath)#market和duke都是按path顺序的，msmt身份按顺序 身份内不顺序
            # imgs=data[0]
            # print('i:',i)#不懂
            # embed_feat = model(imgs)
            # print('输入网络之前：',img.shape)
            embed_feat = extract_feature(model, img, camids, target_view)
            # print('embed_feat.shape:',embed_feat.shape)#256*768 136*768 或者64*768 8*768
            features.append(embed_feat)
            # global_labels.append(i)
            # for fname, output, pid in zip(fnames, outputs, pids):
            #     features[fname] = output
            pid = torch.tensor(pid)
            pids.append(pid)

    print('finishing extracting features')
    # print('len(pids):',len(pids))#203
    features = torch.cat(features, dim=0).numpy()
    pids = torch.cat(pids, dim=0).numpy()
    # print('shape(pids):', pids.shape)#12936
    # print('cat之后的feature：',len(features),len(features[0]))#12936*768
    # print('global_labels长度：',len(global_labels))#51 ???

    # new_features = []

    # for glab in np.unique(global_labels):
    # for idx in range(len(features)):
    #     # idx = np.where(global_labels == glab)[0]
    #
    #     new_features.append(np.mean(features[idx], axis=0))#这样子的话是把一个特征的所有维度平均之后拿去聚类

    # new_features = np.array(new_features)
    # del features
    features = np.array(features)
    # print('features shape:', features.shape)
    # print('features:', features)

    # tensor(-0.0222) tensor(6.8271) tensor(-8.1406)
    # print('features_wonorm:', torch.mean(features_wonorm), torch.max(features_wonorm), torch.min(features_wonorm))  #

    # new_features = new_features / np.linalg.norm(new_features, axis=1, keepdims=True)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    return features, pids