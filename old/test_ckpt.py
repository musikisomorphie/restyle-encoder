# import torch
# from models.stylegan2.model import Generator


# # restyle = '/home/jwu/Experiment/non_IID/rxrx19b_HUVEC_chn0_128/checkpoints/iteration_800000.pt'
# # stylegan = '/home/jwu/Experiment/non_IID/rxrx19b_HUVEC_chn0_128/checkpoints/790000.pt'

# restyle = '/home/jwu/Experiment/non_IID/ham10k_psp_9/checkpoints/iteration_100000.pt'
# stylegan = '/home/jwu/Data/non_IID/encoder/ham10k_tiny/060000.pt'


# decoder = Generator(128, 512, 8, channel_multiplier=2, img_chn=3)
# dec_st = list(decoder.state_dict().keys())
# # print(list(decoder.state_dict().keys()))


# # aa = torch.load(restyle)
# generator = torch.load(stylegan)
# gen_st = list(generator['g_ema'].keys())

# print(dec_st == gen_st)
# # dec = {}
# # for key, v in aa['state_dict'].items():
# #     if 'decoder' in key:
# #         dec[key.replace('decoder.', '')] = v

# # for key, v in bb['g_ema'].items():
# #     print(key, (dec[key] == v).all(), dec[key].dtype, v.dtype)
# #     if 'input' in key:
# #         print(dec[key] - v)




# import torch

# # restyle = '/home/jwu/Experiment/non_IID/rxrx19b_HUVEC_chn0_128/checkpoints/iteration_800000.pt'
# # stylegan = '/home/jwu/Experiment/non_IID/rxrx19b_HUVEC_chn0_128/checkpoints/790000.pt'

# restyle = '/home/jwu/Experiment/non_IID/ham10k_psp_ckpt/checkpoints/iteration_1000.pt'
# stylegan = '/home/jwu/Data/non_IID/encoder/ham10k_tiny/060000.pt'


# aa = torch.load(restyle)
# bb = torch.load(stylegan)
# dec = {}
# for key, v in aa['state_dict'].items():
#     if 'decoder' in key:
#         dec[key.replace('decoder.', '')] = v

# print(list(dec.keys()) == list(bb['g_ema'].keys())) 
# print(dec.keys())

# for key, v in bb['g_ema'].items():
#     if not (dec[key] == v).all():
#         print(key, dec[key].dtype, v.dtype)
#     # print(key, (dec[key] == v).all(), dec[key].dtype, v.dtype)
#     # if 'input' in key:
#     #     print(dec[key] - v)

import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from torchvision.models.resnet import resnet34


resnet_basenet = resnet34()
blocks = [
    resnet_basenet.layer1,
    resnet_basenet.layer2,
    resnet_basenet.layer3,
    resnet_basenet.layer4
]
modules = []
for block in blocks:
    for bottleneck in block:
        modules.append(bottleneck)
body = Sequential(*modules)
print(body)