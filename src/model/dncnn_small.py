import torch
from torch import nn
from torch.nn import init
import pdb

N, k, l = 64, 3, 1

class Net(nn.Module):

    def __init__(self):
        # This line is very important!
        super().__init__()
        self.conv1 = nn.Conv2d(3, N, k, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        m = []
        for _ in range(l):
            m.append(nn.Conv2d(N, N, k, padding=1, bias=False))
            m.append(nn.BatchNorm2d(N, eps=0.0001, momentum=0.95))
            m.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*m)

        self.convf = nn.Conv2d(N, 3, k, padding=1, bias=False)

        # For initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Autograd will keep the history.
        if self.training:
            r = self.relu1(self.conv1(x))
            r = self.seq(r)
            r = self.convf(r)
        else: 
            r = self.forward_chop(x, shave=20, min_size=16000000)

        y = x - r 
        return y

    def forward_chop(self, *args, shave=20, min_size=1600000):
        n_GPUs = 1
        # height, width
        h, w = args[0].size()[-2:]

        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                #pdb.set_trace()
                #y = self.seq(*x)
                y = self.relu1(self.conv1(*x))
                y = self.seq(y)
                y = self.convf(y)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in x_chops[0]:
                print('p.size = ',p.size())
                y = self.forward_chop(p.unsqueeze(0), shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0]

        return y
