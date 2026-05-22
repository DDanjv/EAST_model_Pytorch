import time
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
import numpy as np
print("imported imgDataset.py")
print(f"torch version: {torch.__version__}")

class imgDataset(Dataset):
    def __init__(self, path_imgs, path_labelAndcoords,img_width = 640, img_height = 360):
        super().__init__()
        self.path_imgs = path_imgs
        self.path_labelAndCoords = path_labelAndcoords
        self.img_width = img_width
        self.img_height = img_height

    def __len__(self):
        return len(self.path_imgs)
    
    def __getitem__(self, index):

        #img to tensor
        path_img = self.path_imgs[index]
        img = cv2.imread(path_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.img_width, self.img_height))
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        img = transform(img)

        labelsForImg = []
        coordsForImg = []
        textAndCoord = self.path_labelAndCoords[index]

        #read the text file and parse the labels and coordinates
        with open(textAndCoord, 'r', encoding='utf-8-sig') as file: 
            line = file.readline()
            while line:
                line = line.strip()
                if line:
                    labelAndcoord = line.split(',')
                    labelsForImg.append([i for i in labelAndcoord[8]])
                    coords = [int(float(f) / 2) for f in labelAndcoord[:8]]
                    coordsForImg.append(coords)       
                line = file.readline() 
        # make coordmap for the image
        TrueMap = torch.zeros(1,self.img_height, self.img_width)
        corners = torch.zeros(8,self.img_height, self.img_width)
        for box in coordsForImg:
            # parsing coords
            edges = [
                (box[0:2], box[2:4]), 
                (box[2:4], box[4:6]), 
                (box[4:6], box[6:8]), 
                (box[6:8], box[0:2])  
            ]
            #bounding box
            corners_xy = [(box[j], box[j+1]) for j in range(0, 8, 2)]
            min_x = min(box[0], box[2], box[4], box[6])
            max_x = max(box[0], box[2], box[4], box[6])
            min_y = min(box[1], box[3], box[5], box[7])
            max_y = max(box[1], box[3], box[5], box[7])
            #scan area
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    cnt = 0
                    for edge in edges:
                        (x1,y1), (x2,y2) = edge
                        if (y < y1) != (y < y2) and y2 != y1 and x < x1 + (y - y1) / (y2 - y1) * (x2 - x1):
                            cnt += 1
                    if cnt % 2 == 1:
                        TrueMap[0][y][x] = 1
                    
                    if TrueMap[0][y][x] == 1:  # only for pixels inside the box
                        for k, (cx, cy) in enumerate(corners_xy):
                            corners[k*2][y][x]   = cx - x  # ∆x to corner k
                            corners[k*2+1][y][x] = cy - y  # ∆y to corner k

                            
        '''
        for i in range(len(coordsForImg)):
            for j in range(0, 8, 2):
                x = coordsForImg[i][j]
                y = coordsForImg[i][j+1]
                corners[0][y][x] = 1
        '''

        return [img , 0 , TrueMap, corners]

