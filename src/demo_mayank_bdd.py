from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import pandas as pd
from opts_mayank_demo import opts
from detectors.detector_factory import detector_factory
from datasets.dataset_factory import dataset_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
class_name = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]


def demo(opt):

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  bblabel=[]
  # opt.debug = max(opt.debug, 1)

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    for index,(image_name) in enumerate(image_names):

      ret = detector.run(image_name)
      # if index > 50:
      #     break


      #################################################################33
      resu = ret['results']
      for data in resu:
          print(resu[data])
          for arr in resu[data]:
              print(arr)
          # print(data)

              # $$$$$$$$$$$$$$$$$$$$$$
              xmin = int(arr[0])
              ymin = int(arr[1])
              xmax = int(arr[2])
              ymax = int(arr[3])
              score = float(arr[4])
              width = 1
              height = 1
              # coordinate = [xmin, ymin, xmax, ymax, class_num]
              # object_name=object_name+"_"+light_color
              # object_name = "traffic_light"
              # print(data)
              object_name = class_name[data-1]
              data_label = [image_name, width, height, object_name, xmin, ymin, xmax, ymax,score]
              # data_label = [file_name, width, height, object_name, xmin, ymin, xmax, ymax]
              if not ((xmin == xmax) and (ymin == ymax)):
                  bblabel.append(data_label)
                  print(file_name)
                  # print()
          else:
              print("file_name")





    ############################## #############################################333
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

    columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax','score']
    df = pd.DataFrame(bblabel, columns=columns)
    df.to_csv('centernet_prediction_val.csv', index=False)

if __name__ == '__main__':
  # opt = opts().init()
  opt = opts().parse()
  demo(opt)
