import time
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
import numpy as np

def show_tensor_image(tensor):
    img = tensor.detach().cpu()
    img = (img * 255).byte()
    img_np = img.permute(1, 2, 0).numpy()
    # If the tensor is single channel (grayscale), convert to BGR for display
    if img_np.shape[2] == 1:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('Tensor Image', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

coordmap = torch.zeros(720, 1280)

coords = [[1009,198,1189,130,1193,193,1009,245],
            [1201,120,1279,92,1279,168,1208,189],
            [860,284,892,277,890,309,859,316],
            [892,274,944,259,944,295,892,310],
            [1011,255,1124,230,1125,252,1012,277],
            [1136,325,1267,320,1267,348,1136,352],
            [1133,369,1279,369,1279,413,1134,409],
            [1137,413,1279,419,1278,465,1137,455],
            [1138,459,1279,471,1278,517,1138,500],
            [1136,510,1165,510,1165,529,1136,529],
            [1167,509,1247,524,1245,542,1164,531],
            [1250,522,1278,529,1277,553,1250,546],
            [1136,532,1162,536,1161,559,1137,553],
            [1160,536,1225,546,1227,569,1161,559],
            [1229,545,1254,549,1255,576,1228,570],
            [1139,654,1190,670,1190,696,1137,680],
            [1190,671,1270,698,1210,699,1190,694],
            [0,50,72,87,54,142,0,123],
            [695,332,746,317,744,350,698,358]]

for box in coords:
    # CHANGE: Reordered slices to ensure points follow a perimeter loop
    p1 = box[0:2] # Top Left
    p2 = box[2:4] # Top Right
    p3 = box[4:6] # Bottom Right
    p4 = box[6:8] # Bottom Left
    
    # CHANGE: Connected edges in a circular sequence (1->2, 2->3, 3->4, 4->1)
    # This prevents the "bowtie" crossing effect
    edges = [
        (p1, p2), 
        (p2, p3), 
        (p3, p4), 
        (p4, p1)  
    ]

    # Find the bounding box
    min_x, max_x = min(p1[0], p2[0], p3[0], p4[0]), max(p1[0], p2[0], p3[0], p4[0])
    min_y, max_y = min(p1[1], p2[1], p3[1], p4[1]), max(p1[1], p2[1], p3[1], p4[1])

    # Scan the area
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            cnt = 0
            for edge in edges:
                (x1, y1), (x2, y2) = edge
                # Ray casting logic from reference image
                if (y < y1) != (y < y2):
                    # CHANGE: Added check to prevent division by zero for horizontal lines
                    if y2 != y1: 
                        if x < x1 + (y - y1) / (y2 - y1) * (x2 - x1):
                            cnt += 1
            
            if cnt % 2 == 1: 
                coordmap[y][x] = 1

show_tensor_image(coordmap.unsqueeze(0))