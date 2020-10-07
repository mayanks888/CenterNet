
import numpy as np
import torch
import torch.onnx.utils as onnx
import src.lib.models.networks.pose_dla_dcn as net
# from lib.models.model import create_model, load_model

from collections import OrderedDict
import cv2

model = net.get_pose_net(num_layers=34, heads={'hm': 80, 'wh': 2, 'reg': 2})

# https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md 这里下载的
# 如果下载不了，可以尝试我提供的连接：http://zifuture.com:1000/fs/public_models/ctdet_coco_dla_2x.pth
checkpoint = torch.load(r"../models/ctdet_coco_dla_2x.pth", map_location="cpu")
checkpoint = checkpoint["state_dict"]
change = OrderedDict()
for key, op in checkpoint.items():
    change[key.replace("module.", "", 1)] = op

model.load_state_dict(change)
model.eval()
model.cuda()

input = torch.zeros((1, 3, 512, 512)).cuda()

# 有个已经导出好的模型：http://zifuture.com:1000/fs/public_models/dladcnv2.onnx
onnx.export(model, (input), "dladcnv2.onnx", output_names=["hm", "wh", "reg"], verbose=True)
