import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import tqdm
import pdb
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from model_sidd import Network
from architect import Architect
from data import backbone
from data import noisy
from os import path


parser = argparse.ArgumentParser("sidd")
parser.add_argument('--dataset', type=str, default='SIDD', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--patch_size', type=int, default=41, help='patch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.7, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=1e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--save', type=str, default='darts', help='experiment name')
parser.add_argument('--sub_save', type=str)
parser.add_argument('--pretrained', type=str)
args = parser.parse_args()
seed = 20201010
total_iteration = 0

log_dir = path.join('..', 'experiment', args.save)
if args.sub_save:
    log_dir = path.join(log_dir, args.sub_save)
else:
    log_dir = path.join(log_dir, time.strftime("%Y%m%d-%H%M%S")) 
utils.create_exp_dir(log_dir)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    np.random.seed(seed)
    cudnn.benchmark = True
    cudnn.enabled=True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logging.info("args = %s", args)

    train_data = noisy.NoisyData(
          '../dataset/{}/train/input'.format(args.dataset),
          '../dataset/{}/train/target'.format(args.dataset),
          training=True,
          p=args.patch_size,
    )
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    #print('num_train = ', num_train)
    #print('indices = ', indices)
    #print('split = ', split)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=4)

    #loader_train = DataLoader(
    #    noisy.NoisyData(
    #      '../dataset/{}/train/input'.format(args.dataset),
    #      '../dataset/{}/train/target'.format(args.dataset),
    #      training=True,
    #      p=args.patch_size,
    #    ),
    #    batch_size=args.batch_size,
    #    shuffle=True,
    #    num_workers=4,
    #    pin_memory=True,
    #)
    loader_eval = DataLoader(
        noisy.NoisyData(
          '../dataset/{}/eval/input'.format(args.dataset),
          '../dataset/{}/eval/target'.format(args.dataset),
          training=False,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    writer = tensorboard.SummaryWriter(log_dir)

    # CUDA configuration
    if torch.cuda.is_available():
      device = torch.device('cuda')
      torch.cuda.manual_seed_all(seed)
    else:
      device = torch.device('cpu')

    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    net = Network(args.init_channels, args.layers, criterion)
    net = net.to(device)
    print(net)
    #logging.info(net)
    num_params = sum(p.numel() for p in net.parameters())
    #logging.info('Total parameters = %d', num_params)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(net))
    #pdb.set_trace()

    if args.pretrained is not None:
        ckp = torch.load(args.pretrained)
        model_state = ckp['model']
        logs = net.load_state_dict(model_state, strict=False)
        print('Missing keys:')
        print(logs.missing_keys)
        print('Unexpected keys:')
        print(logs.unexpected_keys)

    #optimizer = torch.optim.SGD(
    #    net.parameters(),
    #    args.learning_rate,
    #    momentum=args.momentum,
    #    weight_decay=args.weight_decay)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    # Set up an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Set up a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5 * args.epochs), int(0.75 * args.epochs)],
        gamma=0.5,
    )

    architect = Architect(net, args)

    def train(epoch, lr):
        global total_iteration
        net.train()
        #tq = tqdm.tqdm(loader_train)
        tq = tqdm.tqdm(train_queue)
        for batch, (x, t) in enumerate(tq):
            x = x.to(device)
            t = t.to(device)
        
            # get a random minibatch from the search queue with replacement
            x_search, t_search = next(iter(valid_queue))
            x_search = x_search.to(device)
            t_search = t_search.to(device)

            architect.step(x, t, x_search, t_search, lr, optimizer, unrolled=args.unrolled)
        
            optimizer.zero_grad()
            #y = net(x)
            r = net(x)
            y = x + r
            loss = F.mse_loss(y, t, reduction='sum')/args.patch_size/args.patch_size
            #loss = F.mse_loss(y, t)
        
            tq.set_description('{:.4f}'.format(loss.item()))
            loss.backward()
            optimizer.step()

            total_iteration += 1
            # Tensorboard batch logging
            if total_iteration % 100 == 0 and total_iteration <= 1000:
                writer.add_images('training_input', utils.quantize(x.cpu()),global_step=total_iteration)
                writer.add_images('training_target',utils.quantize(t.cpu()),global_step=total_iteration)
                writer.add_images('training_output',utils.quantize(y.cpu()),global_step=total_iteration)

            if total_iteration % 10 == 0:
                writer.add_scalar('training_loss', loss.item(), global_step=total_iteration)

    def infer(epoch):
        net.eval()
        avg_loss = 0
        avg_psnr = 0
        with torch.no_grad():
            for idx, (x, t) in enumerate(tqdm.tqdm(loader_eval)):
                x = x.to(device)
                t = t.to(device)

                #y = net(x)
                r = net(x)
                y = x + r
                avg_loss += F.mse_loss(y, t, reduction='sum')/y.size(2)/y.size(3)
                #avg_loss += F.mse_loss(y, t)
                avg_psnr += utils.psnr(y, t)

                y_ = y.detach().cpu()
                y_save = utils.quantize(y_)
                output_dir = path.join(log_dir, 'output')
                os.makedirs(output_dir, exist_ok=True)
                vutils.save_image(y_save, path.join(output_dir, '{:0>2}.png'.format(idx + 1)))

            avg_loss /= len(loader_eval)
            avg_psnr /= len(loader_eval)
            logging.info('Avg. loss: {:.4f} / Avg. PSNR: {:.2f}'.format(
                avg_loss, avg_psnr,
            ))
            # Tensorboard logging for evaluation
            writer.add_scalar('evaluation_loss', avg_loss.item(), global_step=epoch)
            writer.add_scalar('evaluation_psnr', avg_psnr,        global_step=epoch)

            return avg_psnr

    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
    
        genotype = net.genotype()
        logging.info('genotype = %s', genotype)
    
        logging.info(F.softmax(net.alphas_normal, dim=-1))
    
        # training
        train(epoch, lr)
    
        ## validation
        avg_psnr = infer(epoch)
    
        to_save = {}
        to_save['model'] = net.state_dict()
        to_save['optimizer'] = optimizer.state_dict()
        to_save['alphas'] = net.alphas_normal
        to_save['misc'] = avg_psnr
        torch.save(to_save, path.join(log_dir, 'checkpoint_{:0>2}.pt'.format(epoch)))

        scheduler.step()

if __name__ == '__main__':
    main() 

