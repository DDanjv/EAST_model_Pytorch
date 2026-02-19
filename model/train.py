import math
import time
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

def train_model(model, loader_train, loader_val, criterion, optimizer, cycles):

    #setting up model and decive
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # best acc and where to store the model
    best_val_acc = 0.0
    best_model_path = "best_model.pth"

    #to train then test the trainin
    for cycle in range(cycles):
        print(f"cycle: {cycle+1}/{cycles}")
        train_loss, train_acc = loop_helper(model,
                                            dataset_loaded = loader_train, 
                                            device = device, 
                                            optimizer = optimizer, 
                                            criterion = criterion, 
                                            train = True)
        val_loss, val_acc = loop_helper(model,
                                            dataset_loaded = loader_train, 
                                            device = device, 
                                            optimizer = optimizer, 
                                            criterion = criterion, 
                                            train = True)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"model beats old with Val Acc: {best_val_acc:.2f}%, saving model.")
            torch.save(model.state_dict(), best_model_path)
        
    return best_model_path
def loop_helper(model, dataset_loaded, device, optimizer, criterion , train = True):

    #for training or eval
    if train:
        model.train()
    else:
        model.eval()

    #params
    correct = 0
    total = 0
    start = time.time()
    running_loss = 0.0

    #cycles through the photos and coords in each batch
    for imgs, _ , coords in dataset_loaded:
        imgs = imgs.to(device)
        trueMaps = []
        for coordsOfOne in coords:
            coordmap = torch.zeros(360,640)
            for box in coordsOfOne:
                # parsing coords
                edges = [
                    (box[0:2], box[2:4]), 
                    (box[2:4], box[4:6]), 
                    (box[4:6], box[6:8]), 
                    (box[6:8], box[0:2])  
                ]
                #bounding box
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
                            coordmap[y][x] = 1

            trueMaps.append(coordmap)
        #print(trueMaps)
        trueMaps = torch.stack(trueMaps)
        trueMaps = trueMaps.to(device)
        #if training 
        if train:
            optimizer.zero_grad()
        p_score_map, p_geo_map  = model(imgs)
        #compare loactions from model to acc loacltions
        score_loss = balanced_cross_entropy_loss(p_score_map, trueMaps)

        ## need to fix loss cal 
        geo_loss = rbox_loss(p_geo_map, trueMaps)

        total_loss = score_loss + geo_loss 
        #if train 

        if train:
            total_loss.backward()
            optimizer.step()

        # cal loss and accuracy
        with torch.no_grad():
            running_loss += total_loss.item()
            predicted = (p_score_map > 0.5).float()
            correct += (predicted == trueMaps).sum().item()
            total += trueMaps.numel()

    avg_loss = running_loss / len(dataset_loaded)
    accuracy = 100 * correct / total
    print(f"Time taken: {time.time() - start:.2f} seconds")
    print(f"Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

                        # y hat is preds and y* is targets
def balanced_cross_entropy_loss(preds, targets, epsilon=1e-7):

    #print("preds shape: ", preds.shape)
    #print("targets shape: ", targets.shape)

    beta = 1 - torch.mean(targets.float())
    
    preds = torch.clamp(preds, epsilon, 1.0 - epsilon)
    
    return ((-beta * targets * torch.log(preds)) - 
            (1-beta)*(1-targets)*(torch.log(1-preds))).mean()

def rbox_loss(preds, targets, epsilon=1e-7):
    preds = torch.clamp(preds, epsilon, 1.0 - epsilon)
    targets = targets.unsqueeze(1)
    #print("preds shape: ", preds.shape)
    #print("targets shape: ", targets.shape)

    top = preds * targets
    bottom = preds + targets
    topAndBottom = torch.abs(top)/torch.abs(bottom)
    fin = - torch.log(topAndBottom)

    return fin.mean()