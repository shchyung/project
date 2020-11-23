import os
from os import path
import random
import argparse
import importlib

import utils
import collections
from functools import partial
from data import backbone
from data import noisy
from model.dncnn_bnn import BinaryConv

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import numpy as np
import tqdm
import pdb

# Argument parsing
parser = argparse.ArgumentParser('NPEX Project')
parser.add_argument('-d', '--dataset', type=str, default='SIDD')
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-s', '--save', type=str, default='quant')
parser.add_argument('-u', '--sub_save', type=str)
parser.add_argument('-p', '--patch_size', type=int, default=50)
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-m', '--model', type=str, default='dncnn_small')
parser.add_argument('-q', '--qmodel', type=str, default='dncnn_bnn')
parser.add_argument('-t', '--pretrained', type=str)
cfg = parser.parse_args()
seed = 20200922
total_iteration = 0
activations = collections.defaultdict(list)
weights = collections.defaultdict(list)
outputs = collections.defaultdict(list)
#alpha = collections.defaultdict(list)
#beta = collections.defaultdict(list)
#activations = []
#weights = []
#outputs = []
a = []
b = []

def main():
    # Random seed initialization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define your dataloader here
    loader_train = DataLoader(
        noisy.NoisyData(
            '../dataset/{}/train/input_all'.format(cfg.dataset),
            '../dataset/{}/train/target_all'.format(cfg.dataset),
            training=True,
            p=cfg.patch_size,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    loader_eval = DataLoader(
        noisy.NoisyData(
            '../dataset/{}/eval/input_all'.format(cfg.dataset),
            '../dataset/{}/eval/target_all'.format(cfg.dataset),
            training=False,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    log_dir = path.join('..', 'experiment', cfg.save)
    if cfg.sub_save:
        log_dir = path.join(log_dir, cfg.sub_save)

    writer = tensorboard.SummaryWriter(log_dir)

    # CUDA configuration
    #if torch.cuda.is_available():
    #    device = torch.device('cuda')
    #    torch.cuda.manual_seed_all(seed)
    #else:
    #    device = torch.device('cpu')
    device = torch.device('cpu')

    # Make a CNN
    net_module = importlib.import_module('.' + cfg.model, package='model')
    net = net_module.Net()
    net = net.to(device)
    print(net)
    #print('Total parameters = ',sum(p.numel() for p in net.parameters()))

    if cfg.pretrained is not None:
        ckp = torch.load(cfg.pretrained)
        model_state = ckp['model']
        logs = net.load_state_dict(model_state, strict=False)
        print('Missing keys:')
        print(logs.missing_keys)
        print('Unexpected keys:')
        print(logs.unexpected_keys)

    qnet_module = importlib.import_module('.' + cfg.qmodel, package='model')
    qnet = qnet_module.Net()
    qnet = qnet.to(device)
    print(qnet)
    #print('Total parameters = ',sum(p.numel() for p in qnet.parameters()))

    # Set up an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-4)

    # Set up a learning rate scheduler
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5 * cfg.epochs), int(0.75 * cfg.epochs)],
        gamma=0.5,
    )

    def save_layer(name, module, input, output):
	    activations[name].append(input[0].cpu())
	    weights[name].append(module.weight.cpu())
	    outputs[name].append(output.cpu())

    def do_train(epoch: int):
        global total_iteration
        global activations, A
        global weights, W
        global outputs, O
        global alpha, a
        global beta, b
        print('Epoch {}'.format(epoch))
        net.train()
        qnet.train()
        tq = tqdm.tqdm(loader_train)
        for batch, (x, t) in enumerate(tq):
            x = x.to(device)
            t = t.to(device)
            #pdb.set_trace()

            for name, m in net.named_modules():
                if name == 'seq':
                    for num, c in m.named_modules():
                        if type(c) == nn.Conv2d:
                            c.register_forward_hook(partial(save_layer, num))

            with torch.no_grad():
                y = net(x)

                A = [torch.cat(value, 0) for name, value in activations.items()]
                W = [torch.cat(value, 0) for name, value in weights.items()]
                O = [torch.cat(value, 0) for name, value in outputs.items()]

                activations.clear()
                weights.clear()
                outputs.clear()

                for k in range(len(O)):
                	a.append(torch.randn(1, requires_grad=True))
                	b.append(torch.randn(A[k].size()[-2:], requires_grad=True))

            #pdb.set_trace()
            optimizer.zero_grad()

            for name, m in qnet.named_modules():
                if type(m) == BinaryConv:
                    m.register_forward_hook(partial(save_layer, name))

            qy = qnet(x, A, W, a, b)

            AB = [torch.cat(value, 0) for name, value in activations.items()]
            WB = [torch.cat(value, 0) for name, value in weights.items()]
            OB = [torch.cat(value, 0) for name, value in outputs.items()]

            activations.clear()
            weights.clear()
            outputs.clear()

            pdb.set_trace()
            loss = F.mse_loss(O[0], OB[0])

            tq.set_description('{:.4f}'.format(loss.item()))
            loss.backward()
            optimizer.step()

            total_iteration += 1
            # Tensorboard batch logging
            if total_iteration % 100 == 0 and total_iteration <= 1000:
                writer.add_images(
                    'training_input',
                    utils.quantize(x.cpu()),
                    global_step=total_iteration
                )
                writer.add_images(
                    'training_target',
                    utils.quantize(t.cpu()),
                    global_step=total_iteration
                )
                writer.add_images(
                    'training_output',
                    utils.quantize(y.cpu()),
                    global_step=total_iteration
                )

            if total_iteration % 10 == 0:
                writer.add_scalar('training_loss', loss.item(), global_step=total_iteration)

    def do_eval(epoch: int):
        qnet.eval()
        avg_loss = 0
        avg_psnr = 0
        with torch.no_grad():
            for idx, (x, t) in enumerate(tqdm.tqdm(loader_eval)):
                x = x.to(device)
                t = t.to(device)

                y = qnet(x)
                avg_loss += F.mse_loss(y, t, reduction='sum')
                avg_psnr += utils.psnr(y, t)

                # Code for saving image
                # 1 x C x H x W
                # y \in [-1, 1]
                y_ = y.detach().cpu()
                y_save = utils.quantize(y_)
                output_dir = path.join(log_dir, 'output')
                os.makedirs(output_dir, exist_ok=True)
                vutils.save_image(y_save, path.join(output_dir, '{:0>2}.png'.format(idx + 1)))

            avg_loss /= len(loader_eval)
            avg_psnr /= len(loader_eval)
            print('Avg. loss: {:.4f} / Avg. PSNR: {:.2f}'.format(
                avg_loss, avg_psnr,
            ))
            # Tensorboard logging for evaluation
            writer.add_scalar(
                'evaluation_loss',
                avg_loss.item(),
                global_step=epoch
            )
            writer.add_scalar(
                'evaluation_psnr',
                avg_psnr,
                global_step=epoch
            )
            to_save = {}
            to_save['model'] = net.state_dict()
            to_save['optimizer'] = optimizer.state_dict()
            to_save['misc'] = avg_psnr
            torch.save(to_save, path.join(log_dir, 'checkpoint_{:0>2}.pt'.format(epoch)))

    # Outer loop
    for i in range(cfg.epochs):
        do_train(i + 1)
        do_eval(i + 1)
        # Learning rate adjustment
        scheduler.step()

    writer.close()

if __name__ == '__main__':
    main()

