import torch
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
import re
from Processing import process_img
from EAST import EAST
from training import train_model

print("_____________________________:starting testing")

# dataset
training_imgs = [torch.rand(512, 512),torch.rand(512, 512),torch.rand(512, 512)]
training_Labels = ['Shop','Dine','SMRT']
training_coords = [[310,504,524,426,555,517,340,594],[594,398,775,336,801,413,620,474],[501,533,777,431,795,503,518,605]] 
# getting image and processing

print("training_imgs: ",training_imgs, training_imgs[0].shape)
print("training_Labels: ",training_Labels)
print("training_coords: ",training_coords)

print("training_imgs: ",type(training_imgs),type(training_imgs[0]))
print("training_Labels: ",type(training_Labels),type(training_Labels[0]))
print("training_coords: ",type(training_coords),type(training_coords[0]))



#to load model
"""
model = EAST(color_channel=1)
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
"""

"""
if os.getenv('Load_model') == 'True':
    model_path = os.getenv('Model_path')
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
else:
    print("No pre-trained model loaded, starting fresh.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#to train model 
train_model(model, training_imgs, 
                training_Labels, 
                training_coords, 
                batch_size=32, 
                cycles=10)

#to save model

model_save_path = os.getenv('Model_save_path', 'east_model.pth')
torch.save(model.state_dict(), model_save_path)"""