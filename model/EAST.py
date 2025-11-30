import torch
import torchvision.transforms as transforms
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F


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
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1),
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

        # Feature-merging branch
        h4 = self.Feature_merging_4(f4)
        h4_up = F.interpolate(h4, scale_factor=2, mode='bilinear', align_corners=True)
        print("h4 shape: ", h4.shape)
        h4 = torch.cat((h4_up, f3), dim=1)

        h3 = self.Feature_merging_3(h4)
        h3_up = F.interpolate(h3, scale_factor=2, mode='bilinear', align_corners=True)
        print("h3 shape: ", h3.shape)
        h3 = torch.cat((h3_up, f2), dim=1)

        h2 = self.Feature_merging_2(h3)
        h2_up = F.interpolate(h2, scale_factor=2, mode='bilinear', align_corners=True)
        print("h2 shape: ", h2.shape)
        h2 = torch.cat((h2_up, f1), dim=1)

        x = self.Feature_extractor_end(h2)

        # Output layers
        score_map = self.output_start(x)
        score_map = self.output_score_map(score_map)
        geo_map = self.output_score_quad_geometry(score_map)

        print("score_map shape: ", score_map.shape)
        print("geo_map shape: ", geo_map.shape)

        return score_map, geo_map