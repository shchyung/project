import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import pdb

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C):
    super(Cell, self).__init__()

    self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, layers, criterion, steps=1, multiplier=2):
    super(Network, self).__init__()
    self._C = C
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    for i in range(layers):
      #print(C_prev_prev, C_prev, C_curr)
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr)
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.end = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(C_prev, 3, 3, padding=1, bias=False),
    )

    # For initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
          m.bias.data.fill_(0)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    if self.training:
      s0 = s1 = self.stem(input)
      for i, cell in enumerate(self.cells):
        weights = F.softmax(self.alphas_normal, dim=-1)
        s0, s1 = s1, cell(s0, s1, weights)
      r = self.end(s1)
    else:
      r = self.forward_chop(input, shave=self._layers*self._multiplier, min_size=640000)
    #out = input - r     
    #return out
    return r

  def forward_chop(self, input, shave=10, min_size=1600000):
    n_GPUs = 1
    # height, width
    h, w = input.size()[-2:]
    #print(input.size())

    top = slice(0, h//2 + shave)
    bottom = slice(h - h//2 - shave, h)
    left = slice(0, w//2 + shave)
    right = slice(w - w//2 - shave, w)
    x_chops = [torch.cat([
      input[..., top, left],
      input[..., top, right],
      input[..., bottom, left],
      input[..., bottom, right]
    ])]

    y_chops = []
    if h * w < 4 * min_size:
      for i in range(0, 4, n_GPUs):
        x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
        #pdb.set_trace()
        s0 = s1 = self.stem(*x)
        for j, cell in enumerate(self.cells):
          weights = F.softmax(self.alphas_normal, dim=-1)
          s0, s1 = s1, cell(s0, s1, weights)
        y = self.end(s1)
        if not isinstance(y, list): y = [y]
        if not y_chops:
          y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
        else:
          for y_chop, _y in zip(y_chops, y):
            y_chop.extend(_y.chunk(n_GPUs, dim=0))
    else:
      for p in x_chops[0]:
        y = self.forward_chop(p.unsqueeze(0), shave=shave, min_size=min_size)
        if not isinstance(y, list): y = [y]
        if not y_chops: y_chops = [[_y] for _y in y]
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

  def _loss(self, input, target):
    output = self(input)
    return self._criterion(output, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [
      self.alphas_normal,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
    )
    return genotype

