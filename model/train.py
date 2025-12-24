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
    for imgs, coords in dataset_loaded:
        imgs = imgs.to(device)
        coords = coords.to(device)
        #if training 
        if train:
            optimizer.zero_grad()
        #need to return loactions of coners
        outputs = model(imgs)
        #compare loactions from model to acc loacltions
        score_loss, geo_loss = criterion(outputs, coords)
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
