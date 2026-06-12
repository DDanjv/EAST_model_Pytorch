import torch
import torchvision.transforms as transforms
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

class EAST(nn.Module):
    
    def __init__(self, color_channel=1, scale_factor=4):
        super(EAST, self).__init__()
        
        # Define channels explicitly to avoid int(float) truncation bugs
        c0 = int(scale_factor ** 2)      # stem:    4
        c1 = int(scale_factor ** 3)      # stage1:  8
        c2 = int(scale_factor ** 3.5)    # stage2: 11
        c3 = int(scale_factor ** 4)      # stage3: 16
        c4 = int(scale_factor ** 4.5)    # stage4: 22
        cm = int(scale_factor ** 2.5)    # merge output: 5
        cm1 = int(scale_factor ** 2.5)   # 32  first merge
        cm2 = int(scale_factor ** 3)     # 64  second merge  
        cm3 = int(scale_factor ** 3.5)   

        self.Feature_extractor_start = nn.Sequential(
            nn.Conv2d(color_channel, c0, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Feature_extractor_1 = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Feature_extractor_2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Feature_extractor_3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Feature_extractor_4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Feature_merging_h2 = nn.Sequential(
            nn.Conv2d(c4 + c3, cm1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cm1), nn.ReLU(inplace=True),
            nn.Conv2d(cm1, cm1, kernel_size=1),
            nn.Conv2d(cm1, cm1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cm1), nn.ReLU(inplace=True)
        )
        self.Feature_merging_h3 = nn.Sequential(
            nn.Conv2d(cm1 + c2, cm2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cm2), nn.ReLU(inplace=True),
            nn.Conv2d(cm2, cm2, kernel_size=1),
            nn.Conv2d(cm2, cm2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cm2), nn.ReLU(inplace=True)
        )
        self.Feature_merging_h4 = nn.Sequential(
            nn.Conv2d(cm2 + c1, cm3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cm3), nn.ReLU(inplace=True),
            nn.Conv2d(cm3, cm3, kernel_size=1),
            nn.Conv2d(cm3, cm3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cm3), nn.ReLU(inplace=True)
        )
        self.Feature_extractor_end = nn.Sequential(
            nn.Conv2d(cm3, cm3, kernel_size=3, stride=1, padding=1)
        )
        self.output_score_map = nn.Conv2d(cm3, 1, kernel_size=1)
        self.output_score_quad_geometry = nn.Conv2d(cm3, 8, kernel_size=1)
        self.output_score_rbox_geometry = nn.Sequential(
            nn.Conv2d(cm3,4,kernel_size=1),
            nn.ReLU(inplace=True)
        )
    def Initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        check = False
        # Feature extractor
        s = self.Feature_extractor_start(x)
        if(check): print("s shape: ", s.shape)
        F4 = self.Feature_extractor_1(s)
        if(check): print("f1 shape: ", F4.shape)
        F3 = self.Feature_extractor_2(F4)
        if(check): print("f2 shape: ", F3.shape)
        F2 = self.Feature_extractor_3(F3)
        if(check): print("f3 shape: ", F2.shape)
        F1 = self.Feature_extractor_4(F2)
        if(check): print("f4 shape: ", F1.shape)

        #merge and decde

        h1 = F.interpolate(F1,size=(F2.shape[2], F2.shape[3]), mode='bilinear', align_corners=True)
        if(check): print("h1 shape after upsample: ", h1.shape)
        concat1 = torch.cat((h1, F2), dim=1)
        h2 = self.Feature_merging_h2(concat1)
        if(check): print("h2 shape: ", h2.shape)

        h2 = F.interpolate(h2, size=(F3.shape[2], F3.shape[3]), mode='bilinear', align_corners=True)
        if(check): print("h2 shape after upsample: ", h2.shape)
        concat2 = torch.cat((h2, F3), dim=1)
        h3 = self.Feature_merging_h3(concat2)
        if(check): print("h3 shape: ", h3.shape)

        h3 = F.interpolate(h3, size=(F4.shape[2], F4.shape[3]) , mode='bilinear', align_corners=True)
        if(check): print("h3 shape after upsample: ", h3.shape)
        concat3 = torch.cat((h3, F4), dim=1)
        h4 = self.Feature_merging_h4(concat3)
        if(check): print("h4 shape: ", h4.shape)

        h4 = F.interpolate(h4, size=(s.shape[2], s.shape[3]) , mode='bilinear', align_corners=True)
        x_out = self.Feature_extractor_end(h4)
        x_out = F.interpolate(x_out, size=(x.shape[2], x.shape[3]) , mode='bilinear', align_corners=True)
        if(check): print("x shape: ", x_out.shape)

        # Output layers
        '''
        output_layer = self.output_start(x_out)
        output_layer = F.interpolate(output_layer, size = (x_out.shape[2], x_out.shape[3]) , mode='bilinear', align_corners=True)
        '''

        score_map = self.output_score_map(x_out)
        if(check): print("score map shape before sigmoid: ", score_map.shape)
        score_map = torch.sigmoid(score_map)
        score_map = (score_map >= 0.5).float()
        if(check): print("score map shape after sigmoid: ", score_map.shape)

        geo_map = self.output_score_quad_geometry(x_out)
        if(check): print("geo_map : ", geo_map.shape)

        quad_geo_map = self.output_score_rbox_geometry(x_out)
        if(check): print("quad_geo_map : ", quad_geo_map.shape)


        
        return score_map, geo_map, quad_geo_map