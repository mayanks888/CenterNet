from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts_mayank_demo import opts
from detectors.detector_factory import detector_factory
import torch.onnx
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    #
    input_shape = [1, 3, opt.input_res, opt.input_res]
    torch.onnx.export(detector.model, torch.randn(input_shape).cuda(), 'ctdet_coco_dlav0_{}.onnx'.format(opt.input_res),
                      export_params=True)

    # model=detector
    # # model.eval()
    # # model.cuda()
    # input = torch.zeros([1, 3, 512, 512]).cuda()
    # torch.onnx.export(model, input, "ctdet_coco_dla_2x.onnx", verbose=True,
    #             operator_export_type=OperatorExportTypes.ONNX)
    #

if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
#
#
# # from lib.opts import opts
# from lib.models.model import create_model, load_model
# from types import MethodType
# import torch.onnx as onnx
# import torch
# from torch.onnx import OperatorExportTypes
# from collections import OrderedDict
# ## onnx is not support dict return value
# ## for dla34
# def pose_dla_forward(self, x):
#     x = self.base(x)
#     x = self.dla_up(x)
#     y = []
#     for i in range(self.last_level - self.first_level):
#         y.append(x[i].clone())
#     self.ida_up(y, 0, len(y))
#     ret = []  ## change dict to list
#     for head in self.heads:
#         ret.append(self.__getattr__(head)(y[-1]))
#     return ret
# ## for dla34v0
# def dlav0_forward(self, x):
#     x = self.base(x)
#     x = self.dla_up(x[self.first_level:])
#     # x = self.fc(x)
#     # y = self.softmax(self.up(x))
#     ret = []  ## change dict to list
#     for head in self.heads:
#         ret.append(self.__getattr__(head)(x))
#     return ret
# ## for resdcn
# def resnet_dcn_forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)
#
#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)
#     x = self.deconv_layers(x)
#     ret = []  ## change dict to list
#     for head in self.heads:
#         ret.append(self.__getattr__(head)(x))
#     return ret
#
# forward = {'dla':pose_dla_forward,'dlav0':dlav0_forward,'resdcn':resnet_dcn_forward}
#
# opt = opts().init()  ## change lib/opts.py add_argument('task', default='ctdet'....) to add_argument('--task', default='ctdet'....)
# opt.arch = 'dla_34'
# # opt.arch = 'res_18'
# opt.heads = OrderedDict([('hm', 80), ('reg', 2), ('wh', 2)])
# opt.head_conv = 256 if 'dla' in opt.arch else 64
# print(opt)
# model = create_model(opt.arch, opt.heads, opt.head_conv)
# model.forward = MethodType(forward[opt.arch.split('_')[0]], model)
# # load_model(model, '/home/mayank_s/codebase/others/centernet/CenterNet/models/ctdet_coco_dla_2x.pth')
# model.eval()
# model.cuda()
# input = torch.zeros([1, 3, 512, 512]).cuda()
# # onnx.export(model, input, "cool.onnx", verbose=False, operator_export_type=OperatorExportTypes.ONNX)
# onnx.export(model, input, "cool.onnx", verbose=False)
# print("play")
# torch.onnx.export(model, input, 'ctdet_coco_dla_2x.onnx'.format(opt.input_res),
#                   export_params=True)
