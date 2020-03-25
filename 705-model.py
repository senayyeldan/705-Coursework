# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:00:01 2020

@author: Senay Yeldan
"""


#My aim here is creating deeper model by adding extra layers.While adding layers, I am aware of possible problem of overfitting.I tried to make the number of the layers 3, 6, 9 which are the magical numbers according to Tesla.:)





import torch
import torchvision
import torch.nn as nn

#Freeze the parameters on fixed features extractor that is resnet 101 and delete last 2 layers

resnet101 = torchvision.models.resnet101(pretrained=True)
for param in resnet101.parameters():
    param.requires_grad = False
    
modules = list(resnet101.children())[:-2]
new_body_backbone = nn.Sequential(*modules)
#print(new_body_backbone)

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained= True)
model.backbone.body = new_body_backbone



# Added 5 more Conv layers to layer_blocks of the fpn to make it deeper.
new_layerk  = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
for k in range(5):
    model.backbone.fpn.layer_blocks.append(new_layerk)
#print(model)
    
    
#Two changes above work very well.The changes below that I am still working on them.
    
---------------------------------------------------------------------------------------------    
#!!!!!!#Added 2 layers to RPN but I could not solve the problem yet.Problem is getting RPN conv?
for k in range(2):
   model.rpn.head.RPNHead.conv.append(new_layerk) # I just wanted to add 2 conv layers append conv but conv does not attribute append
   


#Added 1 layer to keypoint_head of  keypoint roi pool

model.roi_heads.keypoint_head.KeyPointRCNNHeads
   


























































