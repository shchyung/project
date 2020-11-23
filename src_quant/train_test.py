from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from os import path
import os
import math
import torch
import tqdm

from torch.nn import functional as F
from torchvision import utils as vutils

def quantize(x):
    x = (x + 1) * 127.5
    x = x.clamp(min=0, max=255)
    x = x.round()
    x /= 255
    return x

def psnr(
        x: torch.Tensor,
        y: torch.Tensor,
        luminance: bool = False,
        crop: int = 4) -> float:

    if luminance:
        pass

    diff = x - y    # B x C x H x W
    diff = diff[..., crop:-crop, crop:-crop]
    mse = diff.pow(2).mean().item()
    max_square = 4
    psnr = 10 * math.log10(max_square / mse)

    return psnr

def test(model, val_loader, epoch=0, writer=None, device=torch.device('cpu'), log_dir='.'):
    avg_loss = 0
    avg_psnr = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for idx, (x, t) in enumerate(tqdm.tqdm(val_loader)):
            x = x.to(device)
            t = t.to(device)

            y = model(x)
            avg_loss += F.mse_loss(y, t, reduction='sum')
            avg_psnr += psnr(y, t)
        
            # Code for saving image
            # 1 x C x H x W
            # y \in [-1, 1]
            y_ = y.detach().cpu()
            y_save = quantize(y_)
            output_dir = path.join(log_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)
            vutils.save_image(y_save, path.join(output_dir, '{:0>2}.png'.format(idx + 1)))

        avg_loss /= len(val_loader)
        avg_psnr /= len(val_loader)
        print('Avg. loss: {:.4f} / Avg. PSNR: {:.2f}'.format(
            avg_loss, avg_psnr,
        ))
        # Tensorboard logging for evaluation
        if writer is not None:
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

    return avg_psnr


def train(model, train_loader, optimizer, epoch, writer, device, total_iteration):
    #global total_iteration
    print('Epoch {}'.format(epoch))
    model.train()
    tq = tqdm.tqdm(train_loader)
    for batch, (x, t) in enumerate(tq):
        x = x.to(device)
        t = t.to(device)

        optimizer.zero_grad()
        y = model(x)
        loss = F.mse_loss(y, t, reduction='sum')

        tq.set_description('{:.4f}'.format(loss.item()))
        loss.backward()
        optimizer.step()

        total_iteration += 1
        # Tensorboard batch logging
        if total_iteration % 100 == 0 and total_iteration <= 1000:
            writer.add_images(
                'training_input',
                quantize(x.cpu()),
                global_step=total_iteration
            )
            writer.add_images(
                'training_target',
                quantize(t.cpu()),
                global_step=total_iteration
            )
            writer.add_images(
                'training_output',
                quantize(y.cpu()),
                global_step=total_iteration
            )

        if total_iteration % 10 == 0:
            writer.add_scalar('training_loss', loss.item(), global_step=total_iteration)

    return total_iteration
