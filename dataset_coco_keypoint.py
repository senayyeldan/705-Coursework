import os,sys
import re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import pycocotools
from pycocotools.coco import COCO
import skimage.io as io


# dataset interface takes the ids of the COCO classes
class COCOData(data.Dataset):

     def __init__(self, **kwargs):

        self.stage = kwargs['stage']
        # keep teh class id for the bbbox (1)
        self.coco_classes_ids = kwargs['class_ids']
        self.coco_interface = kwargs['coco_interface']
        # this returns the list of image objects, equal to the number of images of the relevant class(es)
        self.datalist = kwargs['datalist'] 
        # load the list of the image
        self.img_data = self.coco_interface.loadImgs(self.datalist)

     # this method normalizes the image and converts it to Pytorch tensor
     # Here we use pytorch transforms functionality, and Compose them together,
     def transform(self, img):
         # these mean values are for RGB!!
         t_ = transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.ToTensor(),
                             #transforms.Normalize(mean=[0.485, 0.457, 0.407],
                             #                     std=[1,1,1])
                             ])

  
         img = t_(img)
         # need this for the input in the model
         # returns image tensor (CxHxW)
         return img

     # downloadthe image 
     # return rgb image
     def load_img(self, idx):      
       im = np.array(io.imread(self.img_data[idx]['coco_url']))
       im = self.transform(im)
       return im

     def load_label(self, idx): 
         # extract the id of the image, get the annotation ids
         im_id = self.img_data[idx]['id']
         annIds = self.coco_interface.getAnnIds(imgIds = im_id, catIds = [1], iscrowd=None)  
         # get the annotations 
         anns = self.coco_interface.loadAnns(annIds)         
         boxes = []
         ids = []
         keypoints = []
         person_id = 1
         # loop through all objects in the image
         # append id, bbox, extract keypoints and append it too
         for a in anns:
             # this is always going to be 1 (person_id)
             #_id = self._idx[a['category_id']]
             _id = person_id 
             ids.append(_id)             
             _box = a['bbox']
             # bboxes are stored as xmin, ymin, w, h, so convert to xmax, ymax
             box = [_box[0], _box[1], _box[0]+_box[2], _box[1]+_box[3]]
             boxes.append(box)             
             _kp = a['keypoints']
             # reshaped to 17,3: 17 keypoints, (x,y) coords and visibility
             _kp = np.reshape(np.array(_kp), (17,3))
             keypoints.append(_kp)
             print(a)
         # Careful with the data types!!!!
         # Also careful with the variable names!!!
         # If you accidentally use the same name for the object labels and the labs (output of the method) 
         # you get an infinite recursion
         boxes = torch.as_tensor(boxes, dtype = torch.float)
         ids = torch.tensor(ids, dtype=torch.int64)
         keypoints = torch.tensor(keypoints)
         labs = {}
         labs['boxes'] = boxes
         labs['labels'] = ids
         labs['keypoints'] = keypoints
         return labs

     # number of images
     def __len__(self):
         return len(self.datalist)


     # return image + labels (bboxes, labels and keypoints) 
     def __getitem__(self, idx):
         X = self.load_img(idx)
         y = self.load_label(idx) 
         return X,y
