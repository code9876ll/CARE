import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
# lu
import numpy as np
import cv2

def do_train(cfg,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    # if device:
    #     model.to(local_rank)
    #     if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
    #         print('Using {} GPUs for training'.format(torch.cuda.device_count()))
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    # scaler = amp.GradScaler()

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            # optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            # with amp.autocast(enabled=True):
            #     score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
            #     loss = loss_fn(score, feat, target, target_cam)

            score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
            loss = loss_fn(score, feat, target, target_cam)

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()


            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()

            # if (n_iter + 1) % log_period == 0:
            #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
            #                 .format(epoch, (n_iter + 1), len(train_loader),
            #                         loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % 40 ==0 or epoch % 60 ==0 or epoch % 80 ==0 or epoch % checkpoint_period ==0:
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

        scheduler.step(epoch)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, _, _, _, _ = evaluator.compute()


    indices = np.argsort(distmat, axis=1)
    n, m = indices.shape
    print('n, m',n, m)
    print('img_path_list',len(img_path_list))
    indices = indices[:,:30]
    print('indices',indices[:5])
    # print('img_path_list[3369]',img_path_list[5],img_path_list[50],img_path_list[500])
    # read_img = cv2.imread('./data/market1501/bounding_box_test/'+img_path_list[3369])
    # cv2.imwrite('./log_sup/draw_img/test/3369.jpg', read_img)
    for i in [20,200,2000]:
        read_img = cv2.imread('./data/market1501/query/' + img_path_list[i])
        cv2.imwrite('./log_sup/draw_img/test_base/' + 'prob' + str(i) + '_.jpg', read_img)
        # cv2.imwrite('./log_sup/draw_img/test_ABC/' + 'prob' + str(i) + '_.jpg', read_img)
        for j in range(30):
            img_idx_j = indices[i][j]
            img_idx = n + img_idx_j
            read_img = cv2.imread('./data/market1501/bounding_box_test/'+img_path_list[img_idx])
            cv2.imwrite('./log_sup/draw_img/test_base/' + 'gallery' + str(i) + '_' + str(j) + '.jpg', read_img)
            # cv2.imwrite('./log_sup/draw_img/test_ABC/' + 'gallery' + str(i) + '_' + str(j) + '.jpg', read_img)

    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


