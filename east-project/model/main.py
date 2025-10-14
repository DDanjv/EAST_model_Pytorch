import torch
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
import re
from Processing import process_img

print("_____________________________:starting")
load_dotenv()

# dataset
training_imgs = []
training_Labels = []
training_coords = []

# getting coords

directory_train = os.getenv('Directory_train')
print("_____________________________:", directory_train)

coord_file = os.path.join(directory_train, 
                          os.listdir(directory_train)[0])
print("_____________________________:", coord_file)

coord_file = open(coord_file, 'r')
for lines in coord_file:
    coord = lines.strip().split(',')
    coord.remove(coord[0])
    coord = [int(i) for i in coord]
    training_coords.append(coord)

print("training coords: ", len(training_coords))
print("ex: ", training_coords[0])

# getting labels

Labels_file = os.path.join(directory_train, 
                          os.listdir(directory_train)[1])
print("_____________________________:", Labels_file)

Labels_file = open(Labels_file, 'r')
for line in Labels_file:
    label = line.strip().split(',')
    label.remove(label[0])
    label = label[0]
    label = re.sub(r'[^a-zA-Z0-9]', '', label)
    training_Labels.append(label)
print("training labels: ", len(training_Labels))
print("ex: ", training_Labels[0])

# getting image and processing

training_imgs = [process_img(os.path.join(directory_train, f)) 
              for f in os.listdir(directory_train) 
              if f.endswith('.png')]

print("training_imgs: ", len(training_imgs))
print("ex: ", training_imgs[0].shape)

#to load model

#to train model 

#to save model