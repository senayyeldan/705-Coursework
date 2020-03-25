from PIL import Image 
import numpy as np
import time
import os, sys, re
from pycocotools.coco import COCO
import dataset_coco_keypoint
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
#kp = out[0]['keypoints']
# skelten starts at 0

skeleton = np.array([[15, 13],
       [13, 11],
       [16, 14],
       [14, 12],
       [11, 12],
       [ 5, 11],
       [ 6, 12],
       [ 5,  6],
       [ 5,  7],
       [ 6,  8],
       [ 7,  9],
       [ 8, 10],
       [ 1,  2],
       [ 0,  1],
       [ 0,  2],
       [ 1,  3],
       [ 2,  4],
       [ 3,  5],
       [ 4,  6]])

coco_interface_val_kp = COCO("keypoints//person_keypoints_val2017.json")
person_val_cats = coco_interface_val_kp.getCatIds()
im_val_ids = coco_interface_val_kp.getImgIds(catIds = person_val_cats)
coco_dataVal_args = {'datalist':im_val_ids, 'coco_interface':coco_interface_val_kp, 'stage':'val', 'class_ids': [1]}
coco_dataVal = dataset_coco_keypoint.COCOData(**coco_dataVal_args)
coco_dataValloader_args = {'batch_size':1, 'shuffle':True}
coco_datavalloader = data.DataLoader(coco_dataVal, **coco_dataValloader_args)


kprcnn_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
kprcnn_model = kprcnn_model.eval()

#im = Image.open("humans.jpeg")
# convert to tensor
#im_input = torchvision.transforms.ToTensor()(im)

for images in iter(coco_datavalloader):
# return tensor of keypoint predictions, keypoints scores, object bboxes, object scores and labels, sorted in descending order
    test = images[1]
    outputs = kprcnn_model(images[0])
 
    # get keypoints prediction/object and the scores 
    #kp_predictions = outputs[0]['keypoints'].detach().numpy()
    kp_predictions = torch.exp(outputs[0]['keypoints'])
    keypoint_predictions = torch.argmax(kp_predictions, 1)
    scores = outputs[0]['scores'].detach().numpy()
    best_score_th = 0.75
    # get the keypoints for the highest scoring objects
    best_kp = kp_predictions[scores>best_score_th]

# # plot the figure:
# fig, ax = plt.subplots(1,1)
# ax.imshow(im)

# def plot_kp(object_kp_tensor):
#     x= object_kp_tensor[:,0]
#     y= object_kp_tensor[:,1]
#     v= object_kp_tensor[:,2]
#     print(v)
#     for sk in skeleton:
#         if np.all(v[sk])>0:
#            ax.plot(x[sk],y[sk], linewidth=1, color='red')
#         ax.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor='blue', markeredgecolor='k',markeredgewidth=2)

# for _b in best_kp:
#     plot_kp(_b)

# fig.savefig("kprcnn_output.png") 
