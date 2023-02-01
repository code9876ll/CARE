from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
# from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg

# lu
from processor.trainers import ClusterContrastTrainer
import logging

from utils.metrics import R1_mAP_eval

import torch.distributed as dist
from processor.evaluators import extract_features
from utils.faiss_rerank import faiss_compute_jaccard_dist
from sklearn.cluster._dbscan import dbscan
from model.cm import ClusterMemory
from datasets import make_dataloader, update_trainloader
import collections
import time
from torch import nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from sklearn.cluster import DBSCAN

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
    parser.add_argument('--min_samples', type=float, default=4,
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

    # if cfg.MODEL.DIST_TRAIN:
    #     torch.cuda.set_device(args.local_rank)

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

    # if cfg.MODEL.DIST_TRAIN:
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    _, train_loader_normal, val_loader, num_query, _, camera_num, view_num, dataset_train = make_dataloader(cfg, pseudo_labeled_dataset=None)


    num_class = len(dataset_train)
    # print('len(dataset_train):', len(dataset_train))# 12956

    model = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num = view_num)
    model.cuda()
    model = nn.DataParallel(model)

    # loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    # optim
    optimizer = make_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    # train_loader = update_trainloader(cfg, pseudo_labeled_dataset)


    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_func = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model)

    scheduler = create_scheduler(cfg, optimizer)

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    # if device:
    #     model.to(args.local_rank)
    #     if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
    #         print('Using {} GPUs for training'.format(torch.cuda.device_count()))
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        # print('epo:',epoch)
        # cnt = 0
        for n_iter, (img, pid, target_cam, target_view) in enumerate(train_loader):
            # print('n_iter:',n_iter)
            # print('type_n_iter:', type(n_iter))#class int
            # cnt+=1
            # print('cnt:', cnt)
            # print('img.shape:',img.shape)
            # print('vid:', pid)
            # print('target_cam.shape:', target_cam.shape)
            # print('target_view.shape:', target_view.shape)
            optimizer.zero_grad()
            # optimizer_center.zero_grad()
            img = img.to(device)
            target = pid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            # print('img.dtype:', img.dtype)
            # print('target.dtype:', target.dtype)
            # print('target_cam.dtype:', target_cam.dtype)
            # print('target_view.dtype:', target_view.dtype)
            # with amp.autocast(enabled=True):
            #     img = img.half()
            # target = target.to(torch.int32)
            # target_cam = target_cam.to(torch.int32)
            # target_view = target_view.to(torch.int32)

            score, feat = model(img, target, cam_label=target_cam, view_label=target_view )

            loss = loss_func(score, feat, target, target_cam)
            # print('loss.dtype:', loss.dtype)#32
            # loss = loss.half()#16
            # print('loss2.dtype:', loss.dtype)
            # print('feat.shape:',feat.shape)#16 768
            # print('score.shape:', score.shape)#16 751

            # scaler.scale(loss).backward()
            #
            # scaler.step(optimizer)
            # scaler.update()

            loss.backward()
            optimizer.step()

            # if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
            #     for param in center_criterion.parameters():
            #         param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
            #     scaler.step(optimizer_center)
            #     scaler.update()

            # acc
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)


            torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
            # if (n_iter + 1) % log_period == 0:
            #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
            #                 .format(epoch, (n_iter + 1), len(train_loader),
            #                         loss_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        # if cfg.MODEL.DIST_TRAIN:
        #     pass
        # else:
        #     logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
        #                 .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
        #             .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        logger.info("Epoch {} done. Time per batch: {:.3f}[s]".format(epoch, time_per_batch))

        if epoch % checkpoint_period == 0:
            # if cfg.MODEL.DIST_TRAIN:
            #     if dist.get_rank() == 0:
            #         torch.save(model.state_dict(),
            #                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            # else:
            #     torch.save(model.state_dict(),
            #                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            torch.save(model.state_dict(),
                       os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        # if epoch % eval_period == 0:
        if epoch % 5 == 0:
        # if epoch % eval_period == 0 or epoch == 2:
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    camids = camids.to(device)
                    target_view = target_view.to(device)
                    # feat = model(img)
                    feat = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

