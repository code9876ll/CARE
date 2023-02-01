import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset, ImageDataset2, ImageDataset_CB
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler2
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi

from .data import IterLoader

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg, pseudo_labeled_dataset=None):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    source_dataset = __factory[cfg.DATASETS.SOURCE_NAMES](root=cfg.DATASETS.ROOT_DIR)

    if not pseudo_labeled_dataset is None:
        dataset_train=pseudo_labeled_dataset
    else:
        dataset_train=dataset.train
    train_set = ImageDataset(dataset_train, train_transforms)
    # print('okoklulu2')
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    # lu
    source_set_normal = ImageDataset(source_dataset.train, val_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        # if cfg.MODEL.DIST_TRAIN:
        #     print('DIST_TRAIN START')
        #     mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
        #     # data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        #     data_sampler = RandomIdentitySampler_DDP(dataset_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        #     batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
        #     train_loader = torch.utils.data.DataLoader(
        #         train_set,
        #         num_workers=num_workers,
        #         batch_sampler=batch_sampler,
        #         collate_fn=train_collate_fn,
        #         pin_memory=True,
        #     )
        # else:
        #     train_loader = DataLoader(
        #         train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        #         # sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        #         sampler=RandomIdentitySampler(dataset_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        #         num_workers=num_workers, collate_fn=train_collate_fn
        #     )
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            # sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            sampler=RandomIdentitySampler(dataset_train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    # print('okoklulu4')
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    source_loader_normal = DataLoader(
        source_set_normal, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num, dataset.train, \
           source_loader_normal, source_dataset.train

def train_collate_fn2(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    # import cv2
    # import numpy as np
    imgs, true_pid, pids, camids, viewids , r_out, _ = zip(*batch)
    # imgs_ori, imgs, true_pid, pids, camids, viewids, r_out, _ = zip(*batch)
    # imgs_ori = T.ToTensor()(imgs_ori).unsqueeze(0)
    # imgs_ori = cv2.cvtColor(np.array(imgs_ori), cv2.COLOR_RGB2BGR)  #
    true_pid = torch.tensor(true_pid, dtype=torch.int64)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), true_pid, pids, camids, viewids, r_out
    # return torch.stack(imgs_ori, dim=0),torch.stack(imgs, dim=0), true_pid, pids, camids, viewids, r_out

def update_trainloader(cfg, pseudo_labeled_dataset):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])
    ori_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.ToTensor(),
    ])
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_set = ImageDataset2(pseudo_labeled_dataset, train_transforms, ori_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        # if cfg.MODEL.DIST_TRAIN:
        #     print('DIST_TRAIN START')
        #     mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
        #     # data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        #     data_sampler = RandomIdentitySampler_DDP(pseudo_labeled_dataset, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        #     batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
        #     train_loader = torch.utils.data.DataLoader(
        #         train_set,
        #         num_workers=num_workers,
        #         batch_sampler=batch_sampler,
        #         collate_fn=train_collate_fn,
        #         pin_memory=True,
        #     )
        # else:
        #     train_loader = DataLoader(
        #         train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        #         # sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        #         sampler=RandomIdentitySampler(pseudo_labeled_dataset, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        #         num_workers=num_workers, collate_fn=train_collate_fn
        #     )
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            # sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            sampler=RandomIdentitySampler2(pseudo_labeled_dataset, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn2
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn2
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    return train_loader

def train_collate_fn_CB(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs_ori, imgs_w, true_pid, pids = zip(*batch)
    true_pid = torch.tensor(true_pid, dtype=torch.int64)
    pids = torch.tensor(pids, dtype=torch.int64)
    # viewids = torch.tensor(viewids, dtype=torch.int64)
    # camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs_ori, dim=0), torch.stack(imgs_w, dim=0), true_pid, pids

def update_outlierloader(cfg, outlier_dataset):
    # train_transforms_ori = T.Compose([
    #     T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
    #     # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
    #     # T.Pad(cfg.INPUT.PADDING),
    #     # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
    #     T.ToTensor(),
    #     T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    #     # RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    #     # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    # ])
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomGrayscale(p=0.2),
        T.ColorJitter(0.4, 0.4, 0.4, 0.4),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        # T.GaussianBlur(kernel_size=cfg.INPUT.SIZE_TRAIN // 20 * 2 + 1, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])
    num_workers = cfg.DATALOADER.NUM_WORKERS
    # train_set = ImageDataset_CB(outlier_dataset, train_transforms_ori, train_transforms)
    train_set = ImageDataset_CB(outlier_dataset, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        outlier_loader = IterLoader(DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH // 4,
            shuffle=True,
            drop_last=True,
            # sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler2(outlier_dataset, cfg.SOLVER.IMS_PER_BATCH // 4, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn_CB
        ))
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        outlier_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH // 4, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn_CB
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    return outlier_loader

def update_sourceloader(cfg, source_dataset):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])
    ori_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.ToTensor(),
    ])
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_set = ImageDataset2(source_dataset, train_transforms, ori_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = IterLoader(DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH // 4,
            # sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            sampler=RandomIdentitySampler2(source_dataset, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn2
        ))
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH // 4, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn2
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    return train_loader