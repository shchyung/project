from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from os import path
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import argparse

from torch.utils import tensorboard
from torch.utils.data import DataLoader
from train_test import train, test

parser = argparse.ArgumentParser(description='Quantization Training')
parser.add_argument('-d', '--dataset', type=str, default='SIDD')
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-l', '--layer', type=int, default=20)
parser.add_argument('-s', '--save', type=str, default='quant')
parser.add_argument('-u', '--sub_save', type=str)
parser.add_argument('-p', '--patch_size', type=int, default=50)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-t', '--pretrained', type=str)
parser.add_argument('-n', '--num_levels', type=int, default=8)
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
from dncnn_quant import Net
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

# CUDA configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    #torch.cuda.manual_seed_all(seed)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

print('==> Full-precision model accuracy')
from quant_op import Q_ReLU, Q_Conv2d
test(model, testloader, 0, None, device, log_dir + '/full')

for name, module in model.named_modules():
    if isinstance(module, Q_ReLU):
        module.n_lv = args.num_levels
        module.bound = 1
    
    if isinstance(module, (Q_Conv2d)):
        module.n_lv = args.num_levels
        module.ratio = 0.5

print('==> Quantized model accuracy')
from quant_op import Q_ReLU, Q_Conv2d
test(model, testloader, 0, None, device, log_dir + '/quant')

best_psnr = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * args.epoch), int(0.75 * args.epoch)], gamma=0.5,)

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

print('==> Fine-tuned model accuracy')
from quant_op import Q_ReLU, Q_Conv2d
test(model, testloader, 0, None, device, log_dir + '/finetune')

writer.close()
