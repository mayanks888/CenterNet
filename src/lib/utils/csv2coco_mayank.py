#remember the output of json to csv will genreate the bbox in (xmin,ymin,xmax,yman format)
# and from json format it came like ( xmin, ymin , width, height)
import os
import cv2
import pandas as pd
import tqdm
import json
import numpy as np
# csv_path='yolo1.csv'
csv_path='/home/mayank_s/datasets/bdd_bosch_centernet/bosch_bdd_val.csv'
# csv_path='/home/mayank_s/datasets/bdd/training_set/only_csv/front_light_bdd.csv'
root='/home/mayank_s/datasets/bdd_bosch/data/images/train_val'
saving_path = ""
data = pd.read_csv(csv_path)
# mydata = data.groupby('img_name')
mydata = data.groupby(['filename'], sort=True)
# print(data.groupby('class').count())
len_group = mydata.ngroups
mygroup = mydata.groups
###############################################3333
x = data.iloc[:, 0].values
y = data.iloc[:, 4:8].values
z = data.iloc[:, -1].values
##################################################
loop = 0

images = list()
annotations = list()
counter=0
counter2=-1
attr_dict = dict()
attr_dict["categories"] = [
    {"supercategory": "none", "id": 1, "name": "traffic_light"}]
# "person","rider","car","bus","truck","bike","motor", "traffic light","traffic sign","train"
attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
for ind, da1 in enumerate(sorted(mygroup.keys())):
    loop += 1
    print(da1)
    index = mydata.groups[da1].values
    ###########33
    counter += 1
    # if counter > 10:
    #     # break
    # 1
    print(counter)


    ##############3
    da = os.path.join(root, da1)
    # image_scale = cv2.imread(image_path, 1)
    image_scale = cv2.imread(da, 1)
    height = image_scale.shape[0]
    width = image_scale.shape[1]
    print(height, width)
    #######################333
    image = dict()
    image['file_name'] = da1.split("/")[-1]
    print(image['file_name'])
    # image['file_name'] = da1
    image['height'] = height
    image['width'] = width
    image['id'] = counter
    empty_image = True
    ##############################
    for read_index in index:
        counter2 += 1


        # print(index)
        top = (int(y[read_index][0]), int(y[read_index][3]))
        bottom = (int(y[read_index][2]), int(y[read_index][1]))
        # cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
        # cv2.putText(image_scale, str(z[read_index]),(int((y[read_index][0] + y[read_index][2]) / 2), int(y[read_index][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), lineType=cv2.LINE_AA)

        if (y[read_index][2] > width) or y[read_index][3] > height:
            print("something is wrong")

        annotation = dict()
        empty_image = False
        annotation["iscrowd"] = 0
        annotation["image_id"] = image['id']
        x1 = y[read_index][0]
        y1 = y[read_index][1]
        x2 = y[read_index][2]
        y2 = y[read_index][3]
        annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
        annotation['area'] = float((x2 - x1) * (y2 - y1))
        annotation['category_id'] = attr_id_dict["traffic_light"]
        annotation['ignore'] = 0
        annotation['id'] = counter2
        annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        annotations.append(annotation)



    if empty_image:
        continue

    images.append(image)

attr_dict["images"] = images
attr_dict["annotations"] = annotations
attr_dict["type"] = "instances"

cool=attr_dict
print('saving...')


##########################3333
import json
from numpyencoder import NumpyEncoder

out_fn =  'boscd_bdd_val.json'
afile = open(out_fn,'w')
afile.write(json.dumps(attr_dict,cls=NumpyEncoder))
afile.close()
##################################
# json_string = json.dumps(attr_dict)
#
#
# with open(out_fn, "w") as file:
#     file.write(json_string)


