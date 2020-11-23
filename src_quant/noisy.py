import os
from os import path
import glob
import random
import numpy as np

import torch
from torch.utils import data

import tqdm
import imageio


class NoisyData(data.Dataset):

    '''
    A backbone dataset class for the deep image restoration.

    Args:
        dir_input (str): A directory for input images.
        dir_target (str): A directory for target images.
        p (int, optional): Patch size.
        c (int, optional): The number of color channels.
        training (bool, optional): Set False to indicate evaluation dataset.

    '''

    def __init__(self, dir_input, dir_target, p=64, c=3, training=True):

        self.dir_input = dir_input
        self.dir_target = dir_target
        self.p = p
        self.c = c
        self.training = training

        self.img_input = sorted(glob.glob(path.join(dir_input, '*.PNG')))
        self.img_target = sorted(glob.glob(path.join(dir_target, '*.PNG')))
        if len(self.img_input) != len(self.img_target):
            raise IndexError('both lists should have the same lengths.')

        img_input_bin = []
        for img in tqdm.tqdm(self.img_input):
            bin_name = img.replace('PNG', 'bin')
            if not path.isfile(bin_name):
                img_file = imageio.imread(img)
                torch.save(img_file, bin_name)

            img_input_bin.append(bin_name)

        self.img_input = img_input_bin

        img_target_bin = []
        for img in tqdm.tqdm(self.img_target):
            bin_name = img.replace('PNG', 'bin')
            if not path.isfile(bin_name):
                img_file = imageio.imread(img)
                torch.save(img_file, bin_name)

            img_target_bin.append(bin_name)

        self.img_target = img_target_bin

    def __getitem__(self, idx):
        '''
        Get an idx-th input-target pair.

        Args:
            idx (int): An index of the pair.

        Return:
            (C x H x W Tensor): An input image.
            (C x sH x sW Tensor): A target image.
        '''
        idx %= len(self.img_input)

        x = torch.load(self.img_input[idx])
        y = torch.load(self.img_target[idx])

        if self.training:
            h = x.shape[0]
            w = x.shape[1]
    
            oh = y.shape[0]
            ow = y.shape[1]
    
            s = int(oh // h)
    
            py = random.randrange(0, h - self.p + 1)
            px = random.randrange(0, w - self.p + 1)
    
            x = x[py:(py + self.p), px:(px + self.p)]
            y = y[s * py:s * (py +self.p), s * px:s * (px + self.p)]

            do_hflip = random.random() < 0.5
            do_vflip = random.random() < 0.5
            do_rot   = random.random() < 0.5
    
            if do_hflip:
                x = x[:, ::-1]
                y = y[:, ::-1]
    
            if do_vflip:
                x = x[::-1]
                y = y[::-1]

            if do_rot:
                x = np.transpose(x, (1, 0, 2))
                y = np.transpose(y, (1, 0, 2))

        x = np.transpose(x, (2, 0, 1))
        # C x H x W / uint8
        # For efficient memory allocation...
        x = np.ascontiguousarray(x)
        # Now we have torch.FloatTensor [0, 255]
        x = torch.from_numpy(x).float()
        x /= 127.5      # [0, 2]
        x -= 1          # [-1, 1]
    
        y = np.transpose(y, (2, 0, 1))
        y = np.ascontiguousarray(y)
        y = torch.from_numpy(y).float()
        y /= 127.5      # [0, 2]
        y -= 1          # [-1, 1]
    
        return x, y

    def __len__(self):
        '''
        Get the length of the dataset.
        
        Return:
            (int): Total number of the input-target pairs in the dataset.
        '''
        if self.training:
            return 3000
        else:
            #print(len(self.img_input))
            return len(self.img_input)

