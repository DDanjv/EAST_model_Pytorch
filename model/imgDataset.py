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
        img = cv2.resize(img, (640, 360))
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
                    coords = [int(float(f) / 2) for f in labelAndcoord[:8]]
                    coordsForImg.append(coords)         
                line = file.readline()
                #building the img of coords to the be put into as tensor 
                '''canvas = np.zeros((640, 360, 3), dtype="uint8")
                for row in coordsForImg:
                    pts = np.array(row).reshape((-1, 1, 2)).astype(np.int32)
                    color = (0, 255, 0)  
                    cv2.fillPoly(canvas, [pts], color)
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ])
                coordsMap = transform(canvas)'''
                """coordmap = torch.zeros(640,360)
                for textbox in coordsForImg:
                    for i in range(0,len(textbox),2):
                        x = textbox[i]
                        y = textbox[i+1]
                        coordmap[x,y] = 1"""
        '''max_size = max(t.size(0) for t in coordsForImg)
        print(max_size)
        for textbox in coordsForImg:
            pad_size = max_size - textbox.size(0)
            coordsForImg = torch.nn.functional.pad(textbox, (0, pad_size), value=0)'''
        #max_size = max(t[0][0] for t in coordsForImg)
        '''print(max_size)
        for t in coordsForImg:
            diff = max_size - len(coordsForImg[t][0])
            for i in range(0,diff):
                coordsForImg[t].append(torch.tensor([]))'''
        #coordsForImg = torch.stack(coordsForImg, dim=0)                      
        return [img , "labels disabled", coordsForImg]

def custom_collate(batch):
    max_len = max(len(item[2]) for item in batch)

    imgs_alt = []
    labels_alt = []
    coords_alt = []

    for imgs, labels, coords in batch:

        pad_len = max_len - len(coords)
        padded_coords = coords.copy()
        for i in range(pad_len):
            padded_coords.append([0,0,0,0,0,0,0,0])
            
        imgs_alt.append(imgs)
        labels_alt.append(labels)
        coords_alt.append(padded_coords)
    imgs_alt = torch.stack(imgs_alt)
    #wcoords_tensor  = torch.stack(coords_alt)
    return imgs_alt, labels_alt, coords_alt
