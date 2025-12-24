import time
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as transforms

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
        img = cv2.resize(img, (360, 360))
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
                    labelsForImg.append([int(i) for i in labelAndcoord[:8]])
                    coordsForImg.append(labelAndcoord[8])
                line = file.readline()

        return [img , labelsForImg, coordsForImg]
        