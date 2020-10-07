
import numpy as np
import torch
import torch.onnx.utils as onnx
import src.lib.models.networks.pose_dla_dcn as net
from src.lib.models.model import create_model, load_model
from src.lib.opts_mayank import opts

from collections import OrderedDict
import cv2

# model = net.get_pose_net(num_layers=34, heads={'hm': 80, 'wh': 2, 'reg': 2})
model = net.get_pose_net_custom(num_layers=34, heads={'hm': 10, 'wh': 2, 'reg': 2,'hm_tl': 1, 'wh_tl': 2, 'reg_tl': 2})
opt = opts().parse()
optimizer = torch.optim.Adam(model.parameters(), opt.lr)
if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
        model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)


model_state_dict = model.state_dict()
# for key, value in model_state_dict.items():
#     print(key, value)
    # value.requires_grad = True
# for i, j in model_state_dict:
#     print(i)
#     print(j)

# ct = 0
# for i, param,key in enumerate(model.parameters(),model_state_dict):
for  param in model.parameters():
    print(param)
    param.requires_grad = False

req_grad=["model.hm_tl","model.wh_tl","model.reg_tl"]
# for hd in model.reg_tl:
for custom_head in (req_grad):
    for hd in eval(custom_head):
        print(hd.parameters())
        for wt in hd.parameters():
            print(wt)
            wt.requires_grad = True

1