import os
import argparse
import time
import numpy as np
from collections import OrderedDict
import wandb

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from helper.util import accuracy, AverageMeter

from torchvision.datasets import CIFAR100

from models import model_dict

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train SSKD student network.')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--root', type=str, default='./data')
parser.add_argument('--data-portion', type=float, choices=[0.1, 0.25, 1.0, 0.5], default=1.0)

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--milestones', type=int, nargs='+', default=[50, 70, 90])

parser.add_argument('--arch', choices=list(model_dict.keys()), type=str)  # student architecture
parser.add_argument('--wandb-path', type=str, help='in format user/wandbproject/id/.../')
parser.add_argument('--filename', type=str, default='last.pth', help='existing weights to load')
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--key', default=None, help='wandb API key')

args = parser.parse_args()
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

trainset = CIFAR100(args.root, train=True, transform=transform_train)
valset = CIFAR100(args.root, train=False, transform=transform_test)
train_subset = torch.utils.data.Subset(trainset, [i for i in range(int(len(trainset.targets) * args.data_portion))])
print('Num training samples', len(train_subset), 'vs total training available of ', len(trainset))
train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=False)
val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=False)


# exp_path = os.path.dirname(args.weights) + '/model.lincls'
model = model_dict[args.arch](num_classes=100).cuda()
# try:
#     state_dict = torch.load(args.weights)['model']
# except:
#     state_dict = torch.load(args.weights)['state_dict']
wandb.login(key=args.key)
wandb_logger = wandb.init(
    id=args.wandb_path.split('/')[2], resume="must"
)
state_dict = wandb.restore(args.filename, run_path=args.wandb_path)['model']

new_dict = OrderedDict()
for k, v in state_dict.items():
    if 'module.encoder_q.' in k:
        new_dict[k[17:]] = v

if len(new_dict.keys()) > 0:
    del new_dict['fc.weight']
    del new_dict['fc.bias']
    missing_stuff = model.load_state_dict(new_dict, strict=False)
else:
    missing_stuff = model.load_state_dict(state_dict, strict=False)

print('*'*50)
print('MISSING KEYS IN LOAD')
print(missing_stuff)
print('*'*50)

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias', 'classifier.weight', 'classifier.bias', 'linear.weight', 'linear.bias']:
        print(name, 'disabled')
        param.requires_grad = False
    else:
        print(name, 'enabled')

try:
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
except:
    try:
        model.classifier.weight.data.normal_(mean=0.0, std=0.01)
        model.classifier.bias.data.zero_()
    except:
        model.linear.weight.data.normal_(mean=0.0, std=0.01)
        model.linear.bias.data.zero_()
print('*'*50)
model = model.cuda()

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias
optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
criterion = nn.CrossEntropyLoss().cuda()

model.eval()

best_acc = 0.

for epoch in range(args.epoch):

    # train
    loss_record = AverageMeter()
    acc_record = AverageMeter()

    start = time.time()
    for images, targets in train_loader:
        optimizer.zero_grad()
        images = images.cuda()
        targets = targets.cuda()

        fs, preds = model(images, is_feat=True)
        loss = criterion(preds, targets)

        loss.backward()
        optimizer.step()

        loss_record.update(loss.item(), images.shape[0])

    run_time = time.time() - start

    for x, target in val_loader:
        x = x.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(x)

        batch_acc = accuracy(output, target, topk=(1,))[0]
        acc_record.update(batch_acc.item(), x.size(0))

    info = 'lincls_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t loss:{:.5f}\t acc:{:.5f}'.format(
        epoch + 1, args.epoch, run_time, loss_record.avg, acc_record.avg)
    print(info)
    wandb_logger.log({'EVAL/loss': loss_record.avg, 'EVAL/acc': acc_record.avg})
    if acc_record.avg > best_acc:
        best_acc = acc_record.avg
        state_dict = dict(epoch=epoch + 1, state_dict=model.state_dict(), loss=loss_record.avg, acc=acc_record.avg)
        torch.save(state_dict, os.path.join(wandb.run.dir, "model.lincls"))
        wandb.save(os.path.join(wandb.run.dir, "model.lincls"))
    scheduler.step()

