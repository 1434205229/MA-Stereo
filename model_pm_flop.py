import torchvision.models
import torch
from thop import profile
from thop import clever_format
import torch.nn as nn
from models import __models__
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import torchsummary
# model, optimizer
model = __models__['CGI_Stereo'](192)
# device = torch.device('cuda:1')
# model = nn.DataParallel(model, device_ids=[0])
# model.cuda(1)
myinput_left = torch.zeros((1, 3, 960, 512))
myinput_right = torch.zeros((1, 3, 960, 512))
flops, params = profile(model, inputs=(myinput_left, myinput_right))
# flops, params = profile(model.to(device), inputs=myinput_left)
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)
# 179.142G 19.933M
# 467.953G 19.933M
