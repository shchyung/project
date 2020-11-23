from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from os import path
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune
import argparse

from torch.utils import tensorboard
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Full Precision Training')
parser.add_argument('-d', '--dataset', type=str, default='SIDD')
parser.add_argument('-e', '--epoch', type=int, default=10)
parser.add_argument('-l', '--layer', type=int, default=20)
parser.add_argument('-s', '--save', type=str, default='dncnn')
parser.add_argument('-u', '--sub_save', type=str)
parser.add_argument('-p', '--patch_size', type=int, default=50)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-t', '--pretrained', type=str)
parser.add_argument('-pt', '--pretrained_pruned', type=str)
parser.add_argument('-a', '--prune_amount', type=float, default=0.2)
args = parser.parse_args()

# Data
print('==> Preparing data..')
import noisy
trainloader = DataLoader(
    noisy.NoisyData(
        '../dataset/{}/train/input'.format(args.dataset),
        '../dataset/{}/train/target'.format(args.dataset),
        training=True, p=args.patch_size,
    ),
    batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
)
testloader = DataLoader(
    noisy.NoisyData(
        '../dataset/{}/eval/input'.format(args.dataset),
        '../dataset/{}/eval/target'.format(args.dataset),
        training=False,
    ),
    batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
)

log_dir = path.join('.', args.save)
if args.sub_save:
    log_dir = path.join(log_dir, args.sub_save)

writer = tensorboard.SummaryWriter(log_dir)

# Model
print('==> Building model..')
from dncnn import Net
model = Net(args.layer)

print(model)
print('Total parameters = ',sum(p.numel() for p in model.parameters()))

if args.pretrained is not None:
    ckp = torch.load(args.pretrained)
    model_state = ckp['model']
    logs = model.load_state_dict(model_state, strict=False)
    print('Missing keys:')
    print(logs.missing_keys)
    print('Unexpected keys:')
    print(logs.unexpected_keys)

last_layer_num = 2 + 3*(args.layer-2)
last_layer_name = 'seq.' + str(last_layer_num)
for name, module in model.named_modules():
# prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):
        if name == 'seq.0' or name == last_layer_name:
            continue
        else:
            prune.ln_structured(module, name='weight', amount=args.prune_amount, n=2, dim=0)
    
print(dict(model.named_buffers()).keys())  # to verify that all masks exist

if args.pretrained_pruned is not None:
    ckp = torch.load(args.pretrained_pruned)
    model_state = ckp['model']
    logs = model.load_state_dict(model_state, strict=False)
    print('Missing keys:')
    print(logs.missing_keys)
    print('Unexpected keys:')
    print(logs.unexpected_keys)


# CUDA configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    #torch.cuda.manual_seed_all(seed)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

best_psnr = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * args.epoch), int(0.75 * args.epoch)], gamma=0.5,)

from train_test import train, test
total_iteration = 0
for epoch in range(start_epoch, start_epoch + args.epoch):
    total_iteration = train(model, trainloader, optimizer, epoch, writer, device, total_iteration) 
    psnr = test(model, testloader, epoch, writer, device, log_dir)
    scheduler.step()

    if psnr > best_psnr:
        best_psnr = psnr
        to_save = {}
        to_save['model'] = model.state_dict()
        to_save['optimizer'] = optimizer.state_dict()
        to_save['misc'] = psnr
        torch.save(to_save, path.join(log_dir, 'checkpoint_{:0>2}.pt'.format(epoch)))

writer.close()
