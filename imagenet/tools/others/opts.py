import argparse
import seed.models as models


def parse_opt():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='SEED PyTorch Distillation on ImageNet')

    parser.add_argument('--data',
                        metavar='DIR',
                        default='/media/drive2/Data_4TB/imagenet2012',
                        help='path to dataset')

    parser.add_argument('--output',
                        metavar='DIR',
                        default='./output',
                        help='path to output folder')

    parser.add_argument('-a',
                        '--student_arch',
                        metavar='ARCH',
                        default='mobilenetv3_large',
                        choices=model_names,
                        help='student encoder architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18),'
                            'example options: resnet18, efficientnet_b0, mobilenetv3_large')

    parser.add_argument('-k',
                        '--teacher_arch',
                        metavar='ARCH',
                        default='resnet50',
                        choices=model_names,
                        help='teacher encoder architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')

    parser.add_argument('-s',
                        '--teacher_ssl',
                        default='moco',
                        choices=['simclr', 'moco', 'swav'],
                        help='choose the ssl pre-trained method.'
                             ' Currently supporting SimLR, MoCo and SWAV')

    parser.add_argument('-j',
                        '--workers',
                        default=32,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 32)')

    parser.add_argument('--epochs',
                        default=200,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('-b',
                        '--batch-size',
                        default=256,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')

    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.03,
                        type=float,
                        metavar='LR',
                        help='initial learning rate', dest='lr')

    parser.add_argument('--schedule',
                        default=[120, 160],
                        nargs='*',
                        type=int,
                        help='learning rate schedule (when to drop lr by 10x)')

    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum of SGD solver')

    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('-p',
                        '--print-freq',
                        default=1000,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')

    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--distill',
                        default=None,
                        type=str,
                        metavar='PATH',
                        help='path to teacher distillation model.')

    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training.')

    parser.add_argument('--t-temp',
                        default=0.02,
                        type=float,
                        help='softmax temperature for distillation (default: 1e-4)')

    parser.add_argument('--s-temp',
                        default=0.7,
                        type=float,
                        help='softmax temperature for distillation (default: 1e-4)')

    parser.add_argument('--method',
                        default=None,
                        choices=['coss', 'cos', 'dino'],
                        help='Loss method to use for distillation.')

    parser.add_argument('--cos',
                        default=True,
                        type=bool,
                        help='learning rate updating strategy using cosine scheduler or no')

    parser.add_argument("--local_rank",
                        default=0,
                        type=int,
                        help='local rank for DistributedDataParallel')

    parser.add_argument("--distributed",
                        default=False,
                        type=bool,
                        help='use DistributedDataParallel mode or not')

    return parser.parse_args()
