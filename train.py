from utils.logger import setup_logger
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss, make_triplet_loss

import random
import torch
import numpy as np
import os
import argparse
from config import cfg

import logging
from utils.metrics import R1_mAP_eval
from processor.evaluators import extract_features
from utils.faiss_rerank import faiss_compute_jaccard_dist
from sklearn.cluster._dbscan import dbscan
from datasets import make_dataloader, update_trainloader, update_outlierloader, update_sourceloader
import collections
import time
from torch import nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.function import adaptive_instance_normalization as adain

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    # lu
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--eps', type=float, default=0.5,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--min_samples', type=float, default=7,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--k1', type=int, default=20,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))


    _, train_loader_normal, val_loader, num_query, _, camera_num, view_num, dataset_train, \
    source_loader_normal, source_train = make_dataloader(cfg, pseudo_labeled_dataset=None)

    num_class = len(dataset_train)

    model = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num = view_num)
    # load pre-trained model
    model.load_param_finetune(cfg.MODEL.SOURCE_PRETRAIN)

    model.cuda()
    model = nn.DataParallel(model)


    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    loss_mse_meter = AverageMeter()
    loss_neg_meter = AverageMeter()
    loss_s_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)


    k = 1 + 1 #
    for epoch in range(1, epochs + 1):
        # extract feature & generate centroid
        with torch.no_grad():
            features, pids = extract_features(model, train_loader_normal, print_freq=50)
            features = torch.from_numpy(features)

            source_features, source_pids = extract_features(model, source_loader_normal, print_freq=50)
            source_features = torch.from_numpy(source_features)
            source_pids = np.array(source_pids)

            # clustering
            W = faiss_compute_jaccard_dist(features, k1=args.k1, k2=args.k2)
            _, pseudo_labels = dbscan(W, eps=args.eps, min_samples=args.min_samples, metric='precomputed', n_jobs=-1)

            logger.info('  updated_label: num_class= {}, {}/{} images are associated.'
                  .format(pseudo_labels.max() + 1, len(pseudo_labels[pseudo_labels >= 0]), len(pseudo_labels)))  # 235
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]
            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)
        source_centers = generate_cluster_features(source_pids, source_features)


        with torch.no_grad():

            # selecting closer source domain classes
            sour_tar_sim = cluster_features.mm(source_centers.t())
            tar_sim = cluster_features.mm(cluster_features.t())
            source_classes = len(source_centers)
            st_sim, st_sim_idx = torch.sort(sour_tar_sim.view(1, -1), descending=True)
            # top k%
            st_sim_idx = st_sim_idx[:, :int(num_cluster * 0.2)]
            st_sim = st_sim[:, :int(num_cluster * 0.2)]

            st_sim_collect = collections.defaultdict(list)
            for i, (idx, sim) in enumerate(zip(st_sim_idx.squeeze(0), st_sim.squeeze(0))):
                st_sim_collect[idx.item() % source_classes].append(sim)

            st_sim_collect_min = {i: np.mean(st_sim_collect[i]) for i in sorted(st_sim_collect.keys())}

            diag = torch.diag(tar_sim)
            diag = torch.diag_embed(diag)
            tar_sim = tar_sim - diag
            t_sim, t_sim_idx = torch.sort(tar_sim.view(1, -1), descending=True)
            t_sim = t_sim[:, :int(num_cluster * 0.5)]
            # selected source domain classes
            uniq_sim = torch.unique(st_sim_idx % source_classes)
            source_select_class = len(uniq_sim)
            source_class_features = source_centers[uniq_sim]  # 选择筛选出来的源域类中心

            uniqsim2idx = {sim.item(): i for i, sim in enumerate(uniq_sim)}

            # NRSS
            lamb = 0.3
            feat_clus_sim = features.mm(cluster_features.t())
            _, idx1 = torch.sort(feat_clus_sim)
            k_cls_idx = idx1[:,:k]
            # the nearest k centroid
            topk_clus = [torch.index_select(cluster_features, 0, i).unsqueeze(0) for i in k_cls_idx] # 根据上面的索引拿到特征
            topk_clus = torch.cat(topk_clus, dim=0)
            delta_clus = topk_clus[:,1]-topk_clus[:,0]

            # disturbed features
            feat_aug = features + lamb * delta_clus
            # feat_aug clustering
            W = faiss_compute_jaccard_dist(feat_aug, k1=args.k1, k2=args.k2)
            _, pseudo_labels_aug = dbscan(W, eps=args.eps, min_samples=args.min_samples, metric='precomputed', n_jobs=-1)
            logger.info('  updated_label_aug: num_class_aug= {}, {}/{} images are associated.'
                  .format(pseudo_labels_aug.max() + 1, len(pseudo_labels_aug[pseudo_labels_aug >= 0]), len(pseudo_labels_aug)))  # 235
            num_cluster_aug = len(set(pseudo_labels_aug)) - (1 if -1 in pseudo_labels_aug else 0)

            # IOU
            pseudo_labels = torch.from_numpy(pseudo_labels)
            pseudo_labels_aug = torch.from_numpy(pseudo_labels_aug)
            N = pseudo_labels.size(0)
            label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
            label_sim_aug = pseudo_labels_aug.expand(N, N).eq(pseudo_labels_aug.expand(N, N).t()).float()
            R_out = 1 - torch.min(label_sim, label_sim_aug).sum(-1) / torch.max(label_sim, label_sim_aug).sum(-1)

            assert ((R_out.min() >= 0) and (R_out.max() <= 1))
            cluster_R_out = collections.defaultdict(list)
            cluster_img_num = collections.defaultdict(int)
            for i, (r_out, label) in enumerate(zip(R_out, pseudo_labels)):
                cluster_R_out[label.item()].append(r_out.item())
                cluster_img_num[label.item()] += 1
            cluster_R_out_min = []
            for i in sorted(cluster_R_out.keys()):
                if i != -1:
                    cluster_R_out_min.append(min(cluster_R_out[i]))
            cluster_R_out_noins = [iou for iou, num in zip(cluster_R_out_min, sorted(cluster_img_num.keys())) if
                                     cluster_img_num[num] > 1]

            mi_I = torch.min(label_sim, label_sim_aug) - torch.eye(len(pseudo_labels))

            # reliable smples' index
            unchange_lbl, _ = torch.max(mi_I, dim=0)
            unchange_lbl[np.where(pseudo_labels == -1)[0]] = 0
            for i, (r_out, label) in enumerate(zip(R_out, pseudo_labels)):
                r_min = cluster_R_out_min[label.item()]
                # if r_out > r_min:
                if r_out > 0.2: # belta
                    unchange_lbl[i] = 0
            unchange_lbl_ind = np.where(unchange_lbl == 1)[0]
            logger.info('unchange_lbl_ind:{}'.format(len(unchange_lbl_ind)))

        del features


        # update dataset
        pseudo_labeled_dataset = []
        outlier_dataset = []
        source_dataset = []

        for i, ((imgpath, pids, camids, trackid), label, r_out) in enumerate(zip((dataset_train), pseudo_labels, R_out)):
            if label != -1:
                if i in unchange_lbl_ind:
                    pseudo_labeled_dataset.append((imgpath, pids, label.item(), camids, 1, 1))
                else:
                    outlier_dataset.append((imgpath, pids, label.item(), camids, 1, 1))

        for i, (imgpath, pids, camids, trackid) in enumerate(source_train):
            if pids in uniq_sim:
                label = uniqsim2idx[pids] + num_cluster
                source_dataset.append((imgpath, label, label, camids, st_sim_collect_min[pids], 1))

        logger.info('len(pseudo_labeled_dataset):{}'.format(len(pseudo_labeled_dataset)))#1656
        logger.info('len(outlier_dataset):{}'.format(len(outlier_dataset)))
        logger.info('len(source_dataset):{}'.format(len(source_dataset)))

        classifier_weight = torch.cat((cluster_features, source_class_features), dim=0)
        classifier_num = num_cluster + source_select_class
        if cfg.MODEL.JPM:
            model.module.classifier.weight.data[:classifier_num].copy_(F.normalize(cluster_features, dim=1).float().cuda())
            model.module.classifier_1.weight.data[:classifier_num].copy_(F.normalize(cluster_features, dim=1).float().cuda())
            model.module.classifier_2.weight.data[:classifier_num].copy_(F.normalize(cluster_features, dim=1).float().cuda())
            model.module.classifier_3.weight.data[:classifier_num].copy_(F.normalize(cluster_features, dim=1).float().cuda())
            model.module.classifier_4.weight.data[:classifier_num].copy_(F.normalize(cluster_features, dim=1).float().cuda())
        else:
            model.module.classifier.weight.data[:classifier_num].copy_(
                F.normalize(classifier_weight, dim=1).float().cuda())

        # optim
        optimizer = make_optimizer(cfg, model)
        scheduler = create_scheduler(cfg, optimizer)

        # dataloader
        train_loader = update_trainloader(cfg, pseudo_labeled_dataset)
        outlier_loader = update_outlierloader(cfg, outlier_dataset)
        outlier_loader.new_epoch()
        source_loader = update_sourceloader(cfg, source_dataset)
        source_loader.new_epoch()

        # loss_func
        loss_func = make_loss(cfg, num_classes=num_cluster)
        loss_triplet = make_triplet_loss(cfg)

        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        loss_mse_meter.reset()
        loss_neg_meter.reset()
        loss_s_meter.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, (img, _, pid, target_cam, target_view, r_inter) in enumerate(train_loader):
            optimizer.zero_grad()
            # reliable
            img = img.to(device)
            target = pid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            # unreliable
            o_img_ori, o_img_w, _, o_target = outlier_loader.next()
            o_img_ori = o_img_ori.to(device)
            o_img_w = o_img_w.to(device)
            r_inter = torch.tensor(r_inter)
            r_inter = r_inter.to(device)
            # source
            s_img, _, s_pid, _, _, w = source_loader.next()
            s_img = s_img.to(device)
            s_pid = s_pid.to(device)
            w = torch.tensor(w)
            w = w.to(device)

            # reliable forward
            score, feat = model(img, target, cam_label=target_cam, view_label=target_view )

            # style transfer
            B, C = feat.shape
            S = 64
            index = torch.LongTensor(random.sample(range(B), S)).to(device)
            style_img = torch.index_select(img, 0, index)
            t = adain(s_img, style_img)
            # source forward
            s_score, s_feat = model(s_img, s_pid)
            s_score = s_score[:, :classifier_num]
            ada_feat_score, ada_feat = model(t, s_pid)

            # unreliable forward
            o_feat_ori = model(o_img_ori, target, cam_label=target_cam, view_label=target_view, qk=True, k=False)
            o_feat_w = model(o_img_w, target, cam_label=target_cam, view_label=target_view, qk=True, k=True)
            # mse loss
            mse_loss_fn = nn.MSELoss().cuda()
            loss_MSE = mse_loss_fn(o_feat_ori, o_feat_w)  # / o_feat_ori.size(0)

            if isinstance(score, list):
                score = [scor[:, :classifier_num] for scor in score[0:]]
            else:
                score = score[:, :classifier_num]

            loss = loss_func(score, feat, target, r_inter)
            loss += loss_MSE

            s_loss = 0.05 * torch.mean(w * F.cross_entropy(s_score, s_pid, reduction='none'))
            loss_ada = mse_loss_fn(s_feat, ada_feat)
            s_loss += 0.05 * loss_ada

            loss += s_loss

            loss.backward()
            optimizer.step()


            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            loss_mse_meter.update(loss_MSE.item(), img.shape[0])
            # loss_neg_meter.update(loss_neg.item(), img.shape[0])
            loss_s_meter.update(s_loss.item(), img.shape[0])

            torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f},{:.3f},{:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_s_meter.avg, loss_mse_meter.avg,loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s]".format(epoch, time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    camids = camids.to(device)
                    target_view = target_view.to(device)
                    feat = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()
