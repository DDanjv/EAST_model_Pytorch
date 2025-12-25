import time
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms
import numpy as np

class imgDataset(Dataset):
    def __init__(self, path_imgs, path_labelAndcoords):
        super().__init__()
        self.path_imgs = path_imgs
        self.path_labelAndCoords = path_labelAndcoords

    def __len__(self):
        return len(self.path_imgs)
    
    def __getitem__(self, index):

        path_img = self.path_imgs[index]
        img = cv2.imread(path_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (360, 640))
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        img = transform(img)

        labelsForImg = []
        coordsForImg = []
        textAndCoord =self.path_labelAndCoords[index]
        with open(textAndCoord, 'r', encoding='utf-8-sig') as file: 
            line = file.readline()
            while line:
                line = line.strip()
                if line:
                    labelAndcoord = line.split(',')
                    labelsForImg.append([i for i in labelAndcoord[8]])
                    print(labelAndcoord[:8])
                    coords = [float(f) / 2 for f in labelAndcoord[:8]]
                    coordsForImg.append(coords)         
                line = file.readline()
                #building the img of coords to the be put into as tensor 
                canvas = np.zeros((360, 640, 3), dtype="uint8")
                for row in coordsForImg:
                    pts = np.array(row).reshape((-1, 1, 2)).astype(np.int32)
                    color = (0, 255, 0)  
                    cv2.fillPoly(canvas, [pts], color)
                coordsMap = torch.from_numpy(canvas)


        return [img , labelsForImg, coordsMap]
        