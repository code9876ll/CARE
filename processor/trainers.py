from __future__ import print_function, absolute_import
import time
from utils.meter import AverageMeter

# lu
import torch
from torch.cuda import amp
from loss import make_loss
import torch.nn.functional as F

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, cfg, epoch, train_loader, optimizer, logger, scheduler, evaluator, loss_func, num_cluster, print_freq=10):
        # lu
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        scaler = amp.GradScaler()
        device = "cuda"

        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        self.encoder.train()
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

            # 分类器和对比学习
            score, feat = self.encoder(img, target, cam_label=target_cam, view_label=target_view)
            # print('feat:', torch.mean(feat), torch.max(feat), torch.min(feat))
            # # res50
            # # score, feat = model(img)
            # if isinstance(score, list):
            #     score = [scor[:, :num_cluster] for scor in score[0:]]
            # else:
            #     score = score[:, :num_cluster]
            # score_contrast = self.memory(feat, target)
            # loss = loss_func(score, feat, target, target_cam)
            # loss += F.cross_entropy(score_contrast, target)

            # 对比学习
            # feat = self.encoder(img, target, cam_label=target_cam, view_label=target_view)

            loss = loss_func(score, feat, target, target_cam)
            score_con = self.memory(feat, target)
            # print('score_con:',torch.mean(score_con),torch.max(score_con),torch.min(score_con))
            loss_con = F.cross_entropy(score_con, target)
            # print('loss_con:',loss_con)
            loss += 0.3 * loss_con
            # print('loss:', loss)

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

            if (n_iter + 1) % print_freq == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        # if cfg.MODEL.DIST_TRAIN:
        #     pass
        # else:
        #     logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
        #                 .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))




