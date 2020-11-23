import torch
import string

ckp = torch.load('./dncnn7/IP_quant8_prune3/checkpoint_167.pt')
m = ckp['model']
kl = list(m.keys())
for k in kl:
    if 'module' in k:
        k_new = k.replace('module.','')
        m[k_new] = m.pop(k)
ckp['model'] = m
torch.save(ckp, './dncnn7/IP_quant8_prune3/model.pt')
