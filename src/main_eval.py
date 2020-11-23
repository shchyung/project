import os
from os import path
import glob
import argparse
import importlib

import imageio
import numpy as np
import tqdm

import torch
from torch import cuda
import pdb

def main() -> None:
    parser = argparse.ArgumentParser('NPEX Image Restoration Lab')
    parser.add_argument('-i', '--input', type=str, default='example')
    parser.add_argument('-o', '--output', type=str, default='output')
    parser.add_argument('-m', '--model', type=str, default='dncnn')
    parser.add_argument('-p', '--pretrained', type=str)
    cfg = parser.parse_args()

    imgs = sorted(glob.glob(path.join(cfg.input, '*.PNG')))
    os.makedirs(cfg.output, exist_ok=True)

    # CUDA configuration
    if cuda.is_available():
        device = torch.device('cuda')
        #print('cuda is available')
    else:
        device = torch.device('cpu')

    net_module = importlib.import_module('.' + cfg.model, package='model')
    net = net_module.Net()
    net = net.to(device)
    print(net)
    print('Total parameters = ',sum(p.numel() for p in net.parameters()))

    # Implement below
    if cfg.pretrained is not None:
        ckp = torch.load(cfg.pretrained)
        model_state = ckp['model']
        net.load_state_dict(model_state, strict=True)

    # Important!
    net.eval()
    with torch.no_grad():
        for img in tqdm.tqdm(imgs, ncols=80):
            x = imageio.imread(img)
            x = np.transpose(x, (2, 0, 1))
            x = np.ascontiguousarray(x)
            x = torch.from_numpy(x)
            x = x.unsqueeze(0)
            x = x.float()
            x = x.to(device)
            x = x / 127.5 - 1
    
            y = net(x)
            y = y.detach().cpu()
    
            y = 127.5 * (y + 1)
            y = y.round()
            y = y.clamp(min=0, max=255)
            y = y.byte()
            y = y.squeeze(0)
            y = y.cpu()
            y = y.numpy()
            y = np.transpose(y, (1, 2, 0))
    
            name = path.basename(img)
            save_as = path.join(cfg.output, name)
            imageio.imwrite(save_as, y)

    return

if __name__ == '__main__':
    main()
