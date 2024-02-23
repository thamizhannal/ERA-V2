# import packages
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def apply_transformations():
    """
    method that defines required transformation for train and test dataset.
    """
    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1407,), (0.4081,))
        ])
    return train_transforms, test_transforms


def create_train_test_dataset(train_transforms, test_transforms):
    """
    this method downloads train and test data set and applyies corresponding transformations.
    """
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
    return train_data, test_data

def get_correct_pred_count(pPrediction, pLabels):
    """
    A method returns prediction count as label
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def train(model, device, train_loader, optimizer, criterion):
    """
    A method that performs feed forward network training and back propagation
    and records metrics such as train accuracy and loss
    """
    model.train()
    pbar = tqdm(train_loader)
    
    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        
        # Predict
        pred = model(data)
    
        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()
    
        # Backpropagation
        loss.backward()
        optimizer.step()
    
        correct += get_correct_pred_count(pred, target)
        processed += len(data)
    
        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_losses = (train_loss/len(train_loader))
    train_accuracy = (100*correct/processed)
    return train_accuracy, train_losses
    


def test(model, device, test_loader, criterion):
    """
    A method that performs feed forward network testing
    and records metrics such as test accuracy and loss
    """
    model.eval()

    test_loss = 0
    correct = 0
    test_processed = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).sum().item() #, reduction='sum').item()  # sum up batch loss

            correct += get_correct_pred_count(output, target)
            #test_processed += len(data)


    test_loss /= len(test_loader.dataset)
    test_accuracy = (100. * correct /len(test_loader.dataset))
    
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_accuracy, test_loss
    