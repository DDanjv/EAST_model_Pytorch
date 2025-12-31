import time
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
import numpy as np

coordmap = torch.zeros(640,360)

          #top l   top r   bot r   bot l
coords = [[377,117,463,117,465,130,378,130],
        [493,115,519,115,519,131,493,131],
        [374,155,409,155,409,170,374,170],
        [492,151,551,151,551,170,492,170],
        [376,198,422,198,422,212,376,212],
        [494,190,539,189,539,205,494,206],
        [374,1,494,0,492,85,372,86]]

coordGrouped = []
for textbox in coords:
    textboxGroup = []
    for i in range(0,len(textbox),2):
        x = textbox[i]
        y = textbox[i+1]
        textboxGroup.append([x,y])
        coordmap[x,y] = 1
    for j in range(0,len(textboxGroup),2):
        x1 , y1 = textboxGroup[j]
        x2 , y2 = textboxGroup[j+1]
        sl = (y2 - y1) / (x2 - x1)
        

    coordGrouped.append(textboxGroup)


print(coordmap)
print(coordGrouped)
