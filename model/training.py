import time
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

def split_Dataset_and_tensorload(training_imgs, 
                training_coords,
                val_split=0.2):
       dataset = TensorDataset(training_imgs, 
                               training_coords)
       training_size = int((1 - val_split) * len(dataset))
       val_size = len(dataset) - training_size
       training_dataset, val_dataset = torch.utils.data.random_split(dataset, [training_size, val_size])
       return training_dataset, val_dataset

def put_in_dataloader(dataset, batch_size=32):
    dataset_loaded = DataLoader(dataset, batch_size = batch_size)
    return dataset_loaded

def put_in_dataloader_with_bias(dataset, num_classes, batch_size=32):

    #pulls labels
    labels = torch.tensor([label for _, label, _ in dataset])

    #creates weights for each class
    class_counts = torch.bincount(labels, minlength = num_classes)
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[labels]

    #creates sampler and dataloader
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    dataset_loaded = DataLoader(dataset, batch_size = batch_size, sampler = sampler)
    return dataset_loaded

def loop_helper(model, loadedata, device, optimizer=None, criterion=None, train=True):
    if train:
        model.train()
    else:
        model.eval()
    correct = 0
    total = 0
    start = time.time()
    running_loss = 0.0

    for imgs, coords in loadedata:
        imgs = imgs.to(device)
        coords = coords.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(imgs)
        score_loss, geo_loss = criterion(outputs, coords)
        if train:
            score_loss.backward()
            geo_loss.backward()
            optimizer.step()
        running_loss += score_loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += coords.size(0)
        correct += (predicted == coords).sum().item()

    avg_loss = running_loss / len(loadedata)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
    return loss / total, correct / total

def train_model(model,training_imgs, training_Labels, training_coords, batch_size=32, cycles=10):
    # load model 
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # split data into train and validation
    training_dataset, val_dataset = split_Dataset_and_tensorload(training_imgs, 
                                                    training_Labels, 
                                                    training_coords,
                                                    val_split=0.2)
    
    # to put data into data loader

    training_dataset_loaded = put_in_dataloader(training_dataset, batch_size=batch_size)
    val_dataset_loaded = put_in_dataloader(val_dataset, batch_size=batch_size)

    # define optimizer and loss function

    best_val_acc = 0.0
    best_model_path = "best_model.pth"

    for cycle in range(cycles):
        print(f"cycle: {cycle+1}/{cycles}")
        train_loss, train_acc = loop_helper(model,
                                            training_dataset_loaded, 
                                            device, 
                                            optimizer=torch.optim.Adam(model.parameters(), lr=0.001), 
                                            criterion=torch.nn.CrossEntropyLoss(), 
                                            train=True)
        val_loss, val_acc = loop_helper(model,
                                            val_dataset_loaded, 
                                            device, 
                                            criterion=torch.nn.CrossEntropyLoss(), 
                                            train=False)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best model with Val Acc: {best_val_acc:.2f}%, saving model.")
            torch.save(model.state_dict(), best_model_path)

    return best_model_path