import torch

# 1. Simulate dummy IoU values for a batch of 4 bounding boxes
# IoU values range from 0 to 1 (higher is better)
iou_values = torch.tensor([0.85, 0.45, 0.92, 0.31], dtype=torch.float32)

# 2. Step 1: Calculate the log step (-log(IoU))
# Clamping prevents log(0) which would result in NaN/Infinity
eps = 1e-7
log_iou_loss = -torch.log(iou_values + eps)
print("Individual Losses:", log_iou_loss)

# 3. Step 2: Reduce to a single number
# Option A: Mean reduction (Standard/Default)
mean_loss = torch.mean(log_iou_loss)
print("Final Loss (Mean):", mean_loss.item())

# Option B: Sum reduction (Rarely used for final loss)
sum_loss = torch.sum(log_iou_loss)
print("Final Loss (Sum):", sum_loss.item())

import torch.nn.functional as F

x = torch.tensor([-2.0, 0.0, 4.0])
output = x.sigmoid()  # tensor([0.1192, 0.5000, 0.9820])

print(output)
import torch

# 1. Simulate dummy IoU values for a batch of 4 bounding boxes
# IoU values range from 0 to 1 (higher is better)
iou_values = torch.tensor([0.85, 0.45, 0.92, 0.31], dtype=torch.float32)

# 2. Step 1: Calculate the log step (-log(IoU))
# Clamping prevents log(0) which would result in NaN/Infinity
eps = 1e-7
log_iou_loss = -torch.log(iou_values + eps)
print("Individual Losses:", log_iou_loss)

# 3. Step 2: Reduce to a single number
# Option A: Mean reduction (Standard/Default)
mean_loss = torch.mean(log_iou_loss)
print("Final Loss (Mean):", mean_loss.item())

# Option B: Sum reduction (Rarely used for final loss)
sum_loss = torch.sum(log_iou_loss)
print("Final Loss (Sum):", sum_loss.item())

import torch.nn.functional as F

x = torch.tensor([-2.0, 0.0, 4.0])
output = x.sigmoid()  # tensor([0.1192, 0.5000, 0.9820])

print(output)
