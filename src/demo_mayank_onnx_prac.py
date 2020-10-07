import torch
import torchvision
import torch.onnx
# from models.model import create_model, load_model
from src.lib.models.model import create_model, load_model

input = torch.randn(1, 3, 512, 512)
## mobilenetv2_10
# model = create_model('mobilenetv2_10', heads={'hm': 1, 'wh': 2, 'reg': 2}, head_conv=24)
## resnet18
model = create_model('res_18', heads={'hm': 80, 'wh': 2, 'reg': 2, 'hm_tl': 4, 'wh_tl': 2, 'reg_tl': 2}, head_conv=256)
# model = load_model(model, '/home/mario/Projects/Obj_det/CenterNet/exp/ctdet/face_res18/model_best.pth')
##############################################333
ct = 0
for i, param in enumerate(model.parameters()):
    print(param)
    param.requires_grad = False
for child in model.children():
    ct += 1
    print(child)
    print(ct)
if ct < 20:
    for param in child.parameters():
        param.requires_grad = False

for hd in model.reg_tl:
    print(hd.parameters())
    for wt in hd.parameters():
        print(wt)
        wt.requires_grad = True
###################################################
onnx_model_name="../onnx/model.onnx"
torch.onnx.export(model, input, onnx_model_name, input_names=["input"], output_names=["hm","wh","reg",'hm_tl', 'wh_tl', 'reg_tl'])
# {'hm': 3, 'dep': 1, 'rot': 8, 'dim': 3, 'wh': 2, 'reg': 2}