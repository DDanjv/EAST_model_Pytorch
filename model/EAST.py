import torch
import torchvision.transforms as transforms
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

class EAST(nn.Module):
    def __init__(self, color_channel=1):
        super(EAST, self).__init__()
        self.color_channel = color_channel

        # Feature extractor
        self.Feature_extractor_start = nn.Sequential(
            nn.Conv2d(color_channel, 16, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Feature_extractor_1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Feature_extractor_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Feature_extractor_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.Feature_extractor_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Feature-merging branch
        self.Feature_merging_4 = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.Feature_merging_3 = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.Feature_merging_2 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.Feature_extractor_end = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1)
        )

        # Output layers
        self.output_start = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        )
        self.output_score_map = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(4, 1, kernel_size=1, stride=1, padding=0)
        )
        self.output_score_quad_geometry = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=0)
        )

    def Initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # Feature extractor
        s = self.Feature_extractor_start(x)
        print("s shape: ", s.shape)
        f1 = self.Feature_extractor_1(s)
        print("f1 shape: ", f1.shape)
        f2 = self.Feature_extractor_2(f1)
        print("f2 shape: ", f2.shape)
        f3 = self.Feature_extractor_3(f2)
        print("f3 shape: ", f3.shape)
        f4 = self.Feature_extractor_4(f3)
        print("f4 shape: ", f4.shape)

        # the unpooling 

        #first input in to fmb
        h1 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True)
        print("h1 shape unpooled: ", h1.shape)
        print("f3 shape: ", f3.shape)
        concat1 = torch.cat((h1, f3), dim=1)
        print(concat1.shape)
        h2 = self.Feature_merging_4(concat1)

        #second input in to fmb
        print("h2 : ",h2.shape)
        h2 = F.interpolate(h2, size=(f2.shape[2], f2.shape[3]), mode='bilinear', align_corners=True)
        print("h2 shape unpooled: ", h2.shape)
        print("f2 shape: ", f2.shape)
        concat2 = torch.cat((h2, f2), dim=1)
        print(concat2.shape)
        h3 = self.Feature_merging_2(concat2)

        #third input into fmb
        h3 = F.interpolate(h3, size=(f1.shape[2], f1.shape[3]) , mode='bilinear', align_corners=True)
        print("h3 shape unpooled: ", h3.shape)
        print("f1 shape: ", f1.shape)
        concat3 = torch.cat((h3, f1), dim=1)

        x = self.Feature_extractor_end(concat3)

        # Output layers
        output_layer = self.output_start(x)
        output_layer = F.interpolate(output_layer, size = (360, 640) , mode='bilinear', align_corners=True)
        sig = nn.Sigmoid()
        score_map = sig(output_layer)
        score_map = self.output_score_map(score_map)
        score_map = sig(score_map)
        geo_map = self.output_score_quad_geometry(score_map)
        
        #print("score_map shape: ", score_map.shape)
        #print("geo_map tesonr:", geo_map)
        #print("geo_map shape: ", geo_map.shape)

        return score_map, geo_map
    


eee = torch.randn(360, 640)
eee = eee.unsqueeze(0).unsqueeze(0)
model = EAST(color_channel=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
eee = eee.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()
outputs = model(eee)

score_map , geo_map = outputs

print("score shape: ",score_map.shape)
print(score_map)

print("geo map :",geo_map.shape)
print(geo_map)



