#!/usr/bin/env python
import os
import time
import json
import torch.optim
from seed.builder import SEED
from others.builder import COS, COSS, DINO
import torch.nn.parallel
import seed.models as models
import torch.distributed as dist
from tools.opts import parse_opt
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder
from tools.logger import setup_logger
from tools.utils import simclr_aug, mocov1_aug, mocov2_aug, swav_aug, adjust_learning_rate, \
     soft_cross_entropy,  AverageMeter, ValueMeter, ProgressMeter, resume_training, \
     load_simclr_teacher_encoder, load_moco_teacher_encoder, load_swav_teacher_encoder, save_checkpoint, \
     init_wandb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

METHODS = {
    'seed': SEED,
    'cos': COS,
    'coss': COSS,
    'dino': DINO
}


def main(args):
    args.resume = os.path.join(args.output, 'last.pth')
    wandb_logger = init_wandb(args)
    # saving WANDB

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        cudnn.benchmark = True

        # create logger
        logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(),
                              color=False, name="SEED")

        # save the distributed node machine
        logger.info('world size: {}'.format(dist.get_world_size()))
        logger.info('local_rank: {}'.format(args.local_rank))
        logger.info('dist.get_rank(): {}'.format(dist.get_rank()))

    else:
        # create logger
        logger = setup_logger(output=args.output, color=False, name="SEED")
        logger.info('Single GPU mode for debugging.')


    # create model
    logger.info("=> creating student encoder '{}'".format(args.student_arch))
    logger.info("=> creating teacher encoder '{}'".format(args.teacher_arch))

    # some architectures are not supported yet. It needs to be expanded manually.
    assert args.teacher_arch in models.__dict__

    # initialize model object, feed student and teacher into encoders.
    model = METHODS[args.method](student=models.__dict__[args.student_arch],
                                  teacher=models.__dict__[args.teacher_arch],
                                  dim=args.dim,
                                  K=args.queue,
                                  s_temp=args.s_temp,
                                  mlp=args.student_mlp,
                                  t_temp=args.t_temp,
                                  dist=args.distributed,
                                  stu=args.teacher_ssl)

    logger.info(model)

    if args.distributed:
        logger.info('Entering distributed mode.')

        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[args.local_rank],
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=True)
        logger.info('Model now distributed.')
        args.lr_mult = args.batch_size / 256
        args.warmup_epochs = 5
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr_mult * args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        args.lr_mult = 1
        args.warmup_epochs = 5
        model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), args.lr,  momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if os.path.exists(args.output):
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model = resume_training(args, model, optimizer, logger)
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # load the SSL pre-trained teacher encoder into model.teacher
    if args.distill:
        if os.path.isfile(args.distill):
            if args.teacher_ssl == 'moco':
                model = load_moco_teacher_encoder(args, model, logger, distributed=args.distributed)
            elif args.teacher_ssl == 'simclr':
                model = load_simclr_teacher_encoder(args, model, logger, distributed=args.distributed)
            elif args.teacher_ssl == 'swav':
                model = load_swav_teacher_encoder(args, model, logger, distributed=args.distributed)
            logger.info("=> Teacher checkpoint successfully loaded from '{}'".format(args.distill))
        else:
            logger.info("wrong distillation checkpoint.")

    if args.teacher_ssl == 'swav': augmentation = swav_aug
    elif args.teacher_ssl == 'simclr': augmentation = simclr_aug
    elif args.teacher_ssl == 'moco' and args.student_mlp: augmentation = mocov2_aug
    else: augmentation = mocov1_aug

    train_dataset = ImageFolder(os.path.join(args.data, 'train'), transform=augmentation)
    # train_dataset = TSVDataset(os.path.join(args.data, 'train.tsv'), augmentation)
    logger.info('Dataset done.')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        # ensure batch size is dividable by # of GPUs
        assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), \
            'Batch size is not divisible by num of gpus.'

        # create distributed dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    else:
        # create distributed dataloader
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
            drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss, method_time = train(train_loader, model, soft_cross_entropy, optimizer, epoch, args, logger)

        if (args.distributed and dist.get_rank() == 0) or not args.distributed:
            wandb_logger.log({'Train/Loss': loss,
                              'Train/LR': optimizer.param_groups[0]['lr'],
                              'Train/Time':method_time},
                             step=epoch)
            logger.info(f'Epoch: {epoch}\t Loss: {loss}\t S-arch: {args.student_arch}\t Method time: {method_time:.4f}')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student_arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })


def train(train_loader, model, criterion, optimizer, epoch, args, logger):
    batch_time = AverageMeter('Batch Time', ':5.3f')
    data_time = AverageMeter('Data Time', ':5.3f')
    method_time = AverageMeter('Method Time', ':5.4f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = ValueMeter('LR', ':5.3f')
    mem = ValueMeter('GPU Memory Used', ':5.0f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, method_time, lr, losses, mem],
        prefix="Epoch: [{}]".format(epoch))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    mem.update(torch.cuda.max_memory_allocated(device=0) / 1024.0 / 1024.0)

    # switch to train mode
    model.train()

    # make key-encoder at eval to freeze BN
    if args.distributed:
        model.module.teacher.eval()

        # check the sanity of key-encoder
        for name, param in model.module.teacher.named_parameters():
            if param.requires_grad:
                logger.info("====================> Key-encoder Sanity Failed, parameters are not frozen.")

    else:
        model.teacher.eval()

        # check the sanity of key-encoder
        for name, param in model.teacher.named_parameters():
           if param.requires_grad:
                logger.info("====================> Key-encoder Sanity Failed, parameters are not frozen.")

    end = time.time()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for i, (images, _) in enumerate(train_loader):

        if not args.distributed:
            images = images.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        method_start_time = time.time()
        with torch.cuda.amp.autocast(enabled=True):
            if args.method == 'seed':
                logit, label = model(image=images)
                loss = criterion(logit, label)
            else:
                loss = args.beta * model(images)
        method_time.update(time.time()-method_start_time)

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg, method_time.avg


if __name__ == '__main__':
    main(parse_opt())
