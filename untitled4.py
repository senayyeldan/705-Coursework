import os,sys,re
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets as datasets
import torch.utils.data as data



import cv2

import torch.nn.functional as F

from PIL import Image as PILImage
from torchvision import transforms as transforms
# choose which model to use
# this is VGG16 with a final layer with 20 neurons
device = torch.device('cpu')
if torch.cuda.is_available:
    device = torch.device('cuda')

print(device)

# Pascal VOC categories
object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                      'cow', 'diningtable', 'dog', 'horse',
                      'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

# convert list to dict
pascal_voc_classes = {}
for id, name in enumerate(object_categories):
    pascal_voc_classes[name]=id
    
print(pascal_voc_classes, len(pascal_voc_classes))

import os,sys,re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as data 
from PIL import Image as PILImage
from torchvision import transforms as transforms

# this class inherit Pytorch Dataset class
# loads 1 data point:
# 1 image and the vector of labels

class PascalVOC2012Dataset(data.Dataset):

     def __init__(self, **kwargs):

        # Classes from Pascal VOC 2012 dataset, in the correct order without the bgr
        self.voc_classes = kwargs['classes']  
        self.dir = kwargs['dir']
        #print(len(self.dir))
        self.dir_gt = kwargs['dir_gt']
        self.img_max_size = kwargs['img_max_size']
        self.imgs = os.listdir(self.dir)

     # this method normalizes the image and converts it to Pytorch tensor
     # Here we use pytorch transforms functionality, and Compose them together,
     # Convert into Pytorch Tensor and transform back to large values by multiplying by 255
     def transform_img(self, img, img_max_size):
         h,w,c = img.shape
         h_,w_ = img_max_size[0], img_max_size[1]
         img_size = tuple([h_,w_])
         # these mean values are for RGB!!
         t_ = transforms.Compose([
                             transforms.ToPILImage(),              
                             transforms.Resize(img_size),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.407, 0.457, 0.485],
                                                  std=[1,1,1])])
         # need this for the input in the model
         img = 255*t_(img)
         # returns image tensor (CxHxW)
         return img

     # load one image
     # idx: index in the list of images
     def load_img(self, idx):
         im = cv2.imread(os.path.join(self.dir, self.imgs[idx]))
         #print(im)
         im = self.transform_img(im, self.img_max_size)
         return im

     # extract the label vector from the ground truth mask
     def get_label(self, idx):
         # get the file name
         file = self.imgs[idx].split('.')[0]   
         gt_mask = np.array(PILImage.open(os.path.join(self.dir_gt, file+'.png')))
         labels = np.unique(gt_mask)
         ignore = [0,255]
         # reduce label by 1 to squeeze them in [0,19]
         correct_labels = [l-1 for l in labels if not l in ignore]
         lab = torch.zeros(len(self.voc_classes), dtype=torch.float)
         #print(lab, correct_labels)
         for c in correct_labels:
             lab[c] = 1
         #print(lab)
         return lab

     #'magic' method: size of the dataset
     def __len__(self):
         return len(os.listdir(self.dir))        

 
     #'magic' method: iterates through the dataset directory to return the image and its gt
     def __getitem__(self, idx):
     # here you have to implement functionality using the methods in this class to return X (image) and y (its label)
     # X must be dimensionality (3,max_size[1], max_size[0]) if you use VGG16
     # y must be dimensioanlity (self.voc_classes)
        X = self.load_img(idx)
        y = self.get_label(idx) 
        return idx, X,y
    
   ########################## LOAD DATASET#######################
# Image max size must match the VGG16 definition! 
    
data_args= {'classes':pascal_voc_classes, 'img_max_size':(224,224), 'dir':'Images','dir_gt':'GT'}
data_point = PascalVOC2012Dataset(**data_args) 
dataloader_args = {'batch_size':64, 'shuffle':True}  
dataloader = data.DataLoader(datapoint, **dataloader_args )  
   
   
 
my_model = torchvision.models.vgg16(pretrained = True)

print(model.classifier())

param in model.features.parameters()
param.requires_grad = False

model.classifier[-1] = nn.Sequential(
                                     nn.Linear(in_features=4096, out_features=20),
                                     nn.LogSoftmax(dim = 1)
                                     )

# create model and loss function, put on CPU/GPU
my_model.train()
print(my_model) 

loss_function = nn.BCEWithLogitsLoss()

if device=="cuda":
   my_model = my_model.to(device)
   loss_function = loss_function.to(device)
   
opt_args = {'lr' : 1e-5, 'weight_decay' : 1e-3 }
optimizer = torch.optim.SGD(my_model.classifier.parameters(), **opt_args)
epoch = 5

for id, batch in enumerate(dataloader):
        optimizer.zero_grad()
        # this should be size, BxCxHxW, Bx20
        idx, X, y = batch
        print(idx, X.size(), y.size(), len(dataloader))
        if device == "cuda":
           X = X.to(device)
           y = y.to(device) 
        output = my_model(X)
        loss = loss_function(output,y)
        loss.backward()
        optimizer.step()
        total_iter += 1
        total_loss_epoch += loss
    # divide total loss by the number of batches
total_loss_epoch/= len(dataloader)
print("Epoch={0:d}, Loss = {1:.2f}".format(e, total_loss_epoch))










    


