# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os # https://stackoverflow.com/questions/55746872/how-to-limit-number-of-cpus-used-by-a-python-script-w-o-terminal-or-multiproces
#os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
#os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
#os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
#os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1


#nohup  torchrun   --nproc-per-node=8   main_dino_webdataset.py 
#--arch vit_small --data_path /local_storage/crc/crc_extracted_patches_256_level3/  --output_dir /data/csabai-group/crc/crc_dino_vit256-16_
#level3_no_norm_train/  --epochs 100 --batch_size 256 --num_workers 8 --teacher_temp 0.06  --warmup_teacher_temp_epochs 10 > log_train_dino_crc_no_norm_level3.txt & 
#[1] 239011


import argparse
import sys
import datetime
import time
import math
import json
from pathlib import Path
import glob
import h5py
from tqdm import tqdm
import braceexpand

#os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7


import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torch.utils.data.dataset import Dataset

import utils
import past_work.vision_transformer as vits
from past_work.vision_transformer import DINOHead

from tiatoolbox.tools import stainnorm
import webdataset as wds
import warnings
# Convert RuntimeWarning to an exception
warnings.filterwarnings("error", category=RuntimeWarning)


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=1, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


## source: https://webdataset.github.io/webdataset/multinode/
## example: https://github.com/tmbdev-archive/webdataset-examples/blob/master/main-wds.py
## BEST SOURCE AND LATEST WORKING: https://github.com/webdataset/webdataset/blob/main/readme.ipynb
## how to set length: https://github.com/webdataset/webdataset/issues/75

def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    #dataset = datasets.ImageFolder(args.data_path, transform=transform)
    #dataset = CLAM_h5_Dataset_memory(args.data_path, transform=transform, normalize=False)
    #dataset = CLAM_h5_Dataset_disk(args.data_path, transform=transform)
    # data source:
    #urls = "/local_storage/crc/crc_extracted_patches_256_level3_webdataset/dataset_{0000..0091}.tar"
    #urls = "/data/csabaibio/crc/crc_extracted_patches_256_level3_webdataset/dataset_{0000..0064}.tar"
    #urls = "/home/abiricz/sshfs_storage_mount/dataset_{0000..0091}.tar"
    #urls = "/home/abiricz/data/imagenet-1k/data/tar_files_training/dataset_{0100..0900}.tar"
    #urls = "/data/csabai-group/imagenet_tar_training_files/dataset_{0100..0900}.tar"
    urls = '/data/csabai-group/multi_scale_tars/dataset_{0000..0256}.tar' # 2M satellite data
    #urls = "/data/csabai-group/crc/crc_extracted_patches_256_level3_webdataset/dataset_{0000..0091}.tar"
    #urls = "/quicklink/crc/dataset_{0000..0009}.tar"
    #urls = '/home/abiricz/crc/crc_extracted_patches_256_level3_webdataset/dataset_{0000..0055}.tar'
    
    dataset_size = int(2.56*10**6) # THIS CAN BE CALCULATED EXACTLY AT THE BEGINNING OR JUST A GUESS IS FINE !
    ## NOTE: shards can be generated with suffled data -> already shuffled which would be great -> do in preproc !
    dataset_batchnum = int(dataset_size // args.batch_size_per_gpu // torch.distributed.get_world_size() )
    
    dataset = ( wds.WebDataset( urls,
                             resampled=True, # needed for multi gpu training
                             repeat=False, # training loop infinite # WAS TRUE !!
                             shardshuffle=1000, # shuffle shards
                             handler=wds.ignore_and_continue
                             )
                .shuffle(10000)
                .decode("pil")
                .to_tuple("ppm") #.to_tuple('image.jpg') #.to_tuple("ppm")
                .map_tuple(transform)
              )
    
    data_loader = wds.WebLoader( dataset, 
                                 num_workers=args.num_workers, 
                                 batch_size=args.batch_size_per_gpu).with_epoch(dataset_batchnum).with_length(dataset_batchnum)
    
    print(f"Data loaded: there are {len(data_loader)} batches in an epoch.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate) # pretrained True
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False) # pretrained True
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint_82from100epochtrain.pth"), #"dino_deitsmall16_pretrain_full_checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        #data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    #print( 'LOADER LEN INSIDE:', len(data_loader) )
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        #print(len(images[0]), images[0][0], type(images[0]), type(images[0][0]) )
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images[0]] # images variable is in a tuple: (1,) 
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        # If the image is torch Tensor, it is expected to have […, 3, H, W] shape,
        # where … means an arbitrary number of leading dimensions
        
        #ToTensor(): Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
        # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
        # if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
        # or if the numpy.ndarray has dtype = np.uint8

        # In the other cases, tensors are returned without scaling.
        
        flip_and_color_jitter = transforms.Compose([
           # transforms.ToPILImage(), # ADDED HERE
            transforms.RandomHorizontalFlip(p=0.5), # img (PIL Image or Tensor) – Image to be flipped.
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], # img (PIL Image or Tensor) – Image to be flipped.
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2), # img (PIL Image or Tensor) – Image to be flipped.
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) ),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            #transforms.ToPILImage(), # ADDED HERE
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            #transforms.ToPILImage(), # ADDED HERE
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            #transforms.ToPILImage(), # ADDED HERE
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        #print('\n\n\nHEREHERE\n\n\n')
        #print(crops)
        #print('\n\n\nHEREHERE\n\n\n')
        #print(type(image), image.shape)
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


#class CLAM_h5_Dataset_memory(Dataset):
#    def __init__(self, parent_folder, transform=None, normalize=False):
#        self.parent_folder = parent_folder
#        self.transform = transform
#        self.normalize = normalize
#        slide_fp = os.path.join(self.parent_folder, f'*.h5')
#        self.files = np.array( sorted( glob.glob(slide_fp) ) )
#        self.dataset_name = 'imgs'
#        self.num_images = self.get_num_images()
#
#        self.target_image = None
#        self.stain_normalizer = None
#
#        if normalize:
#            print('Normalization applied!')
#            self.target_image = np.load('reference_image_patches.npy')
#            self.stain_normalizer = stainnorm.MacenkoNormalizer()
#            self.stain_normalizer.fit(self.target_image)
#
#        print('NUM IMAGES to load:', self.num_images )
#        self.data = self.load_data()        
#        
#    def get_num_images(self):
#        # Get the total number of images
#        num_images = 0
#        for input_file in self.files:
#            with h5py.File(input_file, 'r') as in_h5:
#                num_images += len(in_h5[self.dataset_name])
#        return num_images
#    
#    def load_data(self):
#        # Define the shape of the final array
#        shape = (self.num_images, 256, 256, 3)
#        
#        # Create an empty numpy array to hold the images
#        data = np.zeros(shape, dtype=np.uint8)
#        data.fill(0)
#
#        # Copy the data from each file to its required place in the final array
#        curr_idx = 0
#        for input_file in tqdm(self.files):
#            print('loaded file:', input_file)
#            with h5py.File(input_file, 'r') as in_h5:
#                num_images = len(in_h5[self.dataset_name])
#
#                ## Normalization:
#                #if self.normalize:
#                #    img_trasformed = np.zeros( (num_images, 256, 256, 3), dtype=np.uint8 )
#                #    print(img_trasformed.shape)
#                #    current_img_array = in_h5[self.dataset_name]
#                #    for current_img_idx in range(num_images):
#                #        #print( img_trasformed.shape, self.stain_normalizer.transform( current_img_array[current_img_idx].copy() ).mean(), current_img_array[current_img_idx].mean() )
#                #        img_trasformed[current_img_idx] = self.stain_normalizer.transform( current_img_array[current_img_idx].copy() )
#                #
#                #    data[curr_idx:curr_idx+num_images] = img_trasformed
#                #    curr_idx += num_images
#                #       
#                #    #image_data = self.stain_normalizer.transform(image_data.copy())
#                #    #print(image_data.min(), image_data.max())
#                #
#                #else:
#                data[curr_idx:curr_idx+num_images] = in_h5[self.dataset_name]
#                curr_idx += num_images
#
#        return data
#    
#    def __len__(self):
#        return self.num_images
#
#    def __getitem__(self, idx):
#        # Get the image data from the final array
#        image_data = self.data[idx]
#
#        if self.normalize:
#            try:
#                image_data = self.stain_normalizer.transform( image_data.copy() )
#            except RuntimeWarning as e:
#                pass # this means no normalization in this case
#                #print(f"Caught a RuntimeWarning during patch normalization, skipping it: {e}") # for debugging
#
#        if self.transform:
#            image_data = self.transform(image_data)
#
#        return image_data # RETURNS PIL IMAGE!
#
#
#
#class CLAM_h5_Dataset_disk(Dataset):
#    def __init__(self, parent_folder, transform=None):
#        self.parent_folder = parent_folder
#        self.transform = transform
#        slide_fp = os.path.join(self.parent_folder, f'*.h5')
#        self.files = np.array( sorted( glob.glob(slide_fp) ) )#[::5]
#        self.dataset_name = None
#        self.file_index_map = self.get_file_idx_map()
#        print('Files initialized!')
#        
#    
#    def get_file_idx_map(self):
#        # Define the paths to the HDF5 files and the dataset name
#        input_files = self.files
#        self.dataset_name = 'imgs'
#
#        # Open the HDF5 files in "r"ead mode and build an index to file map
#        curr_idx = 0
#        file_index_map = {}
#        for input_file in tqdm(input_files):
#            with h5py.File(input_file, 'r') as in_h5:
#                num_images = len(in_h5[self.dataset_name])
#                file_index_map.update({(i+curr_idx): input_file for i in range(num_images)})
#                curr_idx += num_images
#
#        return file_index_map
#
#    
#    def __len__(self):
#        return len(self.file_index_map)
#
#
#    def __getitem__(self, idx):
#        input_file = self.file_index_map[idx]
#
#        # Open the file and get the image data
#        with h5py.File(input_file, 'r') as in_h5:
#            # Get the local index of the image in the file
#            local_idx = idx - next(k for k, v in self.file_index_map.items() if v == input_file)
#
#            # Get the image data from the file
#            image_data = in_h5[self.dataset_name][local_idx]
#            #print(image_data.shape, type(image_data))
#            #image_data = Image.fromarray(image_data) # CONVERT TO PIL HERE FOR TRANSFORMS!
#            
#            if self.transform:
#                image_data = self.transform(image_data)
#        
#        return image_data # RETURNS PIL IMAGE!

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
