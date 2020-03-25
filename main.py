###############################################   LAB 8   ##############################################
################### Script written by Dr Alex Ter-Sarkisov@City, University of London, 2020 ############
##################### DEEP LEARNING CLASSIFICATION, MSC IN ARTIFICIAL INTELLIGENCE #####################
########################################################################################################
import time
import os, sys, re
from pycocotools.coco import COCO
import torch
import torchvision
import dataset_coco_keypoint
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils import data
from hyperparameters import Hyperparam

device = torch.device('cpu')

if torch.cuda.is_available():
   device = torch.device('cuda')

###################### load COCO interface for keypoints, the input is a json file with annotations ####################
coco_interface_train_kp = COCO("keypoints//person_keypoints_train2017.json")
coco_interface_val_kp = COCO("keypoints//person_keypoints_val2017.json")
# just one index: 1 for person 
person_train_cats = coco_interface_train_kp.getCatIds()
person_val_cats = coco_interface_val_kp.getCatIds()
# add background class
# get names of cateogories for keypoints: 
# create labels from ids, no background class
all_keypoints = coco_interface_train_kp.cats[1]['keypoints']
all_keypoints_ids = list(range(len(all_keypoints)))
###############################################
# load ids of images with this class
# Dataset class takes this list as an input and creates data objects 
im_train_ids = coco_interface_train_kp.getImgIds(catIds = person_train_cats)

im_val_ids = coco_interface_val_kp.getImgIds(catIds = person_val_cats)
##############################################
# selected class ids: extract class id from the annotation
coco_dataTrain_args = {'datalist':im_train_ids, 'coco_interface':coco_interface_train_kp, 'stage':'train', 'class_ids': [1]}
coco_dataTrain = dataset_coco_keypoint.COCOData(**coco_dataTrain_args)
coco_dataTrainloader_args = {'batch_size':1, 'shuffle':True}
coco_datatrainloader = data.DataLoader(coco_dataTrain, **coco_dataTrainloader_args)
##############################################
# selected class ids: extract class id from the annotation
coco_dataVal_args = {'datalist':im_val_ids, 'coco_interface':coco_interface_val_kp, 'stage':'val', 'class_ids': [1]}
coco_dataVal = dataset_coco_keypoint.COCOData(**coco_dataVal_args)
coco_dataValloader_args = {'batch_size':1, 'shuffle':True}
coco_datavalloader = data.DataLoader(coco_dataVal, **coco_dataValloader_args)
################### Keypoint R-CNN MODEL ################################################
# 17 classes! 
#        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
#        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
kprcnn_args = {'min_size':256, 'max_size':512}

####################### EVAL OF THE PRETRAINED MODEL #############################
kprcnn_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, **kprcnn_args)
print(kprcnn_model)
kprcnn_model.train()

if device == torch.device('cuda'):
   kprcnn_model = kprcnn_model.to(device)

kprcnn_model.train()

kprcnn_optimizer_pars = {'lr':Hyperparam.lr}
kprcnn_optimizer = optim.Adam(list(kprcnn_model.parameters()),**kprcnn_optimizer_pars)

start_time = time.time()

# add the training code 
for e in range(Hyperparam.epochs):
   epoch_loss = 0
   total_image = 0
   for _batch in coco_datatrainloader:
      kprcnn_optimizer.zero_grad()
      total_image += 1
      X, y = _batch
      if device == torch.device('cuda'):
         X =  X.to(device)
         y["labels"] = y["labels"].to(device)
         y["boxes"] = y["boxes"].to(device)
         y["keypoints"] = y["keypoints"].to(device)
         # keypoints = torch.tensor(keypoints, dtype=torch.float)
      images = [im for im in X]
      targets = []
      lab = {}
      lab['boxes'] = y["boxes"].squeeze_(0)
      lab['labels'] = y["labels"].squeeze_(0)
      keypoints = y["keypoints"].squeeze_(0)
      keypoints = torch.tensor(keypoints, dtype=torch.float)
      lab['keypoints'] = keypoints.to(device)
      targets.append(lab)
      if len(targets) > 0:
         loss = kprcnn_model(images, targets)
         total_loss = 0
         for k in loss.keys():
            total_loss += loss[k]
         
         temp = total_loss.clone().detach().cpu()
         epoch_loss += temp.numpy()
         total_loss.backward()
         kprcnn_optimizer.step()
   epoch_loss = epoch_loss / total_image
   print("loss in epoch {0:d} = {1:3f}".format(e, epoch_loss))

end_time = time.time()

print("Training took {0:.1f}".format(end_time-start_time))

