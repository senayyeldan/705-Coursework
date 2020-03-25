# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:00:01 2020

@author: Senay Yeldan
"""


#Deeper Model for Keypoint Detection


import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import KeypointRCNN

#Change fixed features extractor as resnet 101 and freeze parameters 

resnet101 = torchvision.models.resnet101(pretrained=True)
for param in resnet101.parameters():
    param.requires_grad = False
    
modules = list(resnet101.children())[:-2] #it takes all layers except for last two ones
new_backbone_body = nn.Sequential(*modules)
#print(new_backbone_body)

model = torchvision.models.detection.keypointrcnn_resnet50_fpn()
model.backbone.body = new_backbone_body
#print(model)


# Added 5 more Conv layers to layer_blocks of the fpn to make 
new_layerk  = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
for k in range(5):
    model.backbone.fpn.layer_blocks.append(new_layerk)
#print(model)
    
backbone = model.backbone
#print(backbone)


#Changes on RPN
#I have got anchor generator`s values from Alex's notes
    
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
#Added two conv layers more to head of RPN 
new_rpn_head=nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                     nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),#NEW
                     nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),#NEW
                     nn.Conv2d(256, 3, kernel_size = (1,1), stride = (1,1)),#cls_logits
                     nn.Conv2d(256, 12, kernel_size =(1,1), stride =(1,1))#bbox_pred
                     )

box_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                         output_size=14,
                                                         sampling_ratio=2)
#put the pieces together
deeper_model = KeypointRCNN(backbone,
                            num_classes=2,
                            rpn_anchor_generator=anchor_generator,
                            rpn_head = new_rpn_head,
                            box_roi_pool = box_roi_pooler,
                            keypoint_roi_pool = keypoint_roi_pooler)

print(deeper_model)











    
    












    







    
   
   
   
   
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   

























































