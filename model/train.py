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
            coordmap = torch.zeros(640,360)
            for box in coordsOfOne:

                # parsing coords 
                topL = box[0:2]
                topR = box[2:4]
                botL = box[4:6]
                botR = box[6:8]

                #geting edges
                topSlope = int((topL[1]-topR[1])/(topL[0]-topR[0]))
                botSlope = int((botL[1]-botR[1])/(botL[0]-botR[0]))
                leftSlope = int((botL[0]-topL[0])/(botL[1]-topL[1]))
                rightSlope = int((botR[0]-topR[0])/(botR[1]-topR[1]))

                #Find the bounding box
                boundingX = [topL[0], topR[0], botL[0], botR[0]]
                boundingY = [topL[1], topR[1], botL[1], botR[1]]
                min_x, max_x = min(boundingX), max(boundingX)
                min_y, max_y = min(boundingY), max(boundingY)

                # Scan the area
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        if (topSlope > y and botSlope < y and leftSlope < x and rightSlope > x):
                            coordmap[y][x] = 1

            trueMaps.append(coordmap)
        trueMaps = torch.stack(trueMaps)
        trueMaps = trueMaps.to(device)
        #if training 
        if train:
            optimizer.zero_grad()
        #need to return loactions of coners
        score_map, geo_map  = model(imgs)
        #compare loactions from model to acc loacltions
        '''
            cv2.fillPoly(gt_score_map, [shrunk_box.astype(np.int32)], 1)
            loss = balanced_cross_entropy(predicted_score_map, gt_score_map) 
        
        '''
        ## need to fix loss cal 
        score_loss = criterion(score_map, coords)

        ## need to fix loss cal 
        geo_loss = criterion(geo_map, coords)
        #if train 
        if train:
            score_loss.backward()
            geo_loss.backward()
            optimizer.step()
        running_loss += score_loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += coords.size(0)
        correct += (predicted == coords).sum().item()

    avg_loss = running_loss / len(dataset_loaded)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
