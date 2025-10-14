import numpy as np
import torch
import torchvision.transforms as transforms
import os
import cv2


def process_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (360, 360))
    transform = transforms.Compose([
        transforms.ToTensor(),])
    img = transform(img)
    return img

