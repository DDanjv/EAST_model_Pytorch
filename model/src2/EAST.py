import torch
from torchvision import models
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler 

class VGG16_FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16_FeatureExtractor, self).__init__()
        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.model = vgg16(weights=weights) 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        #self.models.vgg16(pretrained=True)

    def Feature_extractor_1_vgg16(self, x):
        return self.model.features[0:5](x)
    
    def Feature_extractor_2_vgg16(self, x):
        return self.model.features[5:10](x)

    def Feature_extractor_3_vgg16(self, x):
        return self.model.features[10:17](x)
     
    def Feature_extractor_4_vgg16(self, x):
        return self.model.features[17:24](x)
    
    def Feature_extractor_5_vgg16(self, x):
        return self.model.features[24:31](x)
    
    def forward(self, x): # use if neededs
        layer_1 = self.Feature_extractor_1_vgg16(x)
        layer_2 = self.Feature_extractor_2_vgg16(layer_1)
        layer_3 = self.Feature_extractor_3_vgg16(layer_2)
        layer_4 = self.Feature_extractor_4_vgg16(layer_3)
        layer_5 = self.Feature_extractor_5_vgg16(layer_4)
        
        return layer_5 

class EAST(nn.Module):
    
    def __init__(self, color_channel=3, scale_factor=4):
        super(EAST, self).__init__()
        self.vgg16_model = VGG16_FeatureExtractor(pretrained=True) 
        print("Using VGG16 as feature extractor for EAST model")
        # Define channels explicitly to avoid int(float) truncation bugs
        c0 = int(scale_factor ** 2)      # stem:    16
        c1 = int(scale_factor ** 3)      # stage1:  64
        c2 = int(scale_factor ** 3.5)    # stage2: 128
        c3 = int(scale_factor ** 4)      # stage3: 256
        c4 = int(scale_factor ** 4.5)    # stage4: 512
        cm = int(scale_factor ** 2.5)    # merge output: 32
        cm1 = int(scale_factor ** 2.5)   # 32  first merge
        cm2 = int(scale_factor ** 3)     # 64  second merge  
        cm3 = int(scale_factor ** 3.5)   # 128
        #print(" ,",c0," ,",c1," ,",c2," ,",c3," ,",c4," ,",cm," ,",cm1," ,",cm2," ,",cm3)

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
            #c4 + c3
            nn.Conv2d((cm*cm1), cm1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cm1), nn.ReLU(inplace=True),
            nn.Conv2d(cm1, cm1, kernel_size=1),
            nn.Conv2d(cm1, cm1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cm1), nn.ReLU(inplace=True)
        )
        self.Feature_merging_h3 = nn.Sequential(
            #cm1 + c2
            nn.Conv2d(cm1 + 256, cm2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cm2), nn.ReLU(inplace=True),
            nn.Conv2d(cm2, cm2, kernel_size=1),
            nn.Conv2d(cm2, cm2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cm2), nn.ReLU(inplace=True)
        )
        self.Feature_merging_h4 = nn.Sequential(
            #cm2 + c1
            nn.Conv2d(cm2 + 128, cm3, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(cm3,5,kernel_size=1)
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
        '''s = self.Feature_extractor_start(x)
        if(check): print("s shape: ", s.shape)
        F4 = self.Feature_extractor_1(s)
        if(check): print("f1 shape: ", F4.shape)
        F3 = self.Feature_extractor_2(F4)
        if(check): print("f2 shape: ", F3.shape)
        F2 = self.Feature_extractor_3(F3)
        if(check): print("f3 shape: ", F2.shape)
        F1 = self.Feature_extractor_4(F2)
        if(check): print("f4 shape: ", F1.shape)'''

        #print(x.shape)
        s = self.vgg16_model.Feature_extractor_1_vgg16(x)
        if(check): print("s shape: ", s.shape)
        F4 = self.vgg16_model.Feature_extractor_2_vgg16(s)
        if(check): print("F4 shape: ", F4.shape)
        F3 = self.vgg16_model.Feature_extractor_3_vgg16(F4)
        if(check): print("F3 shape: ", F3.shape)
        F2 = self.vgg16_model.Feature_extractor_4_vgg16(F3)
        if(check): print("F2 shape: ", F2.shape)
        F1 = self.vgg16_model.Feature_extractor_5_vgg16(F2)
        if(check): print("F1 shape: ", F1.shape)
        
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

        x_out = self.Feature_extractor_end(h4)
        if(check): print("x shape: ", x_out.shape)

        # Output layers
        x_out = F.interpolate(x_out, size = (x.shape[2], x.shape[3]) , mode='bilinear', align_corners=True)

        score_map = self.output_score_map(x_out)
        if(check): print("score map shape before sigmoid: ", score_map.shape)
        score_map = torch.sigmoid(score_map)
        if(check): print("score map shape after sigmoid: ", score_map.shape)

        geo_map = self.output_score_quad_geometry(x_out)
        if(check): print("geo_map : ", geo_map.shape)

        quad_geo_map = self.output_score_rbox_geometry(x_out)
        distances = torch.sigmoid(quad_geo_map[:, :4, :, :])
        angle = (torch.pi / 2) * torch.tanh(quad_geo_map[:, 4:5, :, :])
        quad_geo_map = torch.cat([distances, angle], dim=1)
        if(check): print("quad_geo_map : ", quad_geo_map.shape)

        return score_map, geo_map, quad_geo_map