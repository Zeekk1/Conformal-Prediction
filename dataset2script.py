"""This file just contains the code for training the model efficiently

The saved and trained model is also provided in the folder

"""


import torchvision.models as models 
import torch.nn as nn   
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
from glob import glob
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Dataset paths
BASE_PATH = r"C:\Users\User\Desktop\Dissertation\Code\dataset2\COVID-19_Radiography_Dataset"
"""COVID_PATH = os.path.join(BASE_PATH, "COVID", "images")
NORMAL_PATH = os.path.join(BASE_PATH, "Normal", "images")
VIRAL_PNEUMONIA_PATH = os.path.join(BASE_PATH, "Viral Pneumonia", "images")"""
COVID_PATH = r"C:\Users\User\Desktop\Dissertation\Code\dataset2\COVID-19_Radiography_Dataset\COVID\images"
NORMAL_PATH= r"C:\Users\User\Desktop\Dissertation\Code\dataset2\COVID-19_Radiography_Dataset\Normal\images"
VIRAL_PNEUMONIA_PATH=r"C:\Users\User\Desktop\Dissertation\Code\dataset2\COVID-19_Radiography_Dataset\Viral Pneumonia\images"



def load_image_paths():
    """Load all image paths from COVID, Normal, and Viral Pneumonia folders"""
    covid_images = glob(os.path.join(COVID_PATH, "*.png")) 
    normal_images = glob(os.path.join(NORMAL_PATH, "*.png"))
    viral_pneumonia_images = glob(os.path.join(VIRAL_PNEUMONIA_PATH, "*.png"))
    
    return covid_images, normal_images, viral_pneumonia_images


# Image transform for ResNet-18 
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
def get_train_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),   
        transforms.RandomRotation(5),             
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
#neeed to change this in the main the transform function

class CovidDataset(Dataset):
    def __init__(self, df, transform, class_roots):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_roots = class_roots

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])
        img_name = row['Image Index']
        
        img_path = os.path.join(self.class_roots[label], img_name)
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)

def paths_to_df(paths, labels):
    """Convert image paths and labels to DataFrame"""
    return pd.DataFrame({
        'Image Index': [Path(p).name for p in paths],
        'label': labels
    })


def create_advanced_model(num_classes):
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  
        nn.Linear(in_features, 512),  
        nn.BatchNorm1d(512), 
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model



def train_model(model, train_loader, val_loader, device, num_epochs=20):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
   
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
        
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 
                             'Acc': f'{100*train_correct/train_total:.2f}%'})
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
            print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
  
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
            
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        # Saving the best model from the training loop
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_covid_model.pth')
    return model, train_losses, val_accuracies
            
  
def main():
    # Loading the  data
    covid_paths, normal_paths, viral_pneumonia_paths = load_image_paths()
    
    # Creating labels: 0=Normal, 1=COVID, 2=Viral Pneumonia
    normal_labels = [0] * len(normal_paths)
    covid_labels = [1] * len(covid_paths)
    viral_pneumonia_labels = [2] * len(viral_pneumonia_paths)
    
  
    all_images = normal_paths + covid_paths + viral_pneumonia_paths
    all_labels = normal_labels + covid_labels + viral_pneumonia_labels

    #Data split is Training 50: 25 Calibration:25 test
    X_temp, X_test, y_temp, y_test = train_test_split(
    all_images, all_labels, test_size=0.25, random_state=42, stratify=all_labels)

    X_train, X_cal, y_train, y_cal = train_test_split(
    X_temp, y_temp, test_size=0.333, random_state=42, stratify=y_temp)

    eval_transform = get_eval_transform()

    
    train_transform = get_train_transform()
    eval_transform = get_eval_transform()
    
    class_roots = {
        0: os.path.join(BASE_PATH, "Normal", "images"),
        1: os.path.join(BASE_PATH, "COVID", "images"),
        2: os.path.join(BASE_PATH, "Viral Pneumonia", "images")
    }
    
    # Create datasets
    train_df = paths_to_df(X_train, y_train)
    cal_df = paths_to_df(X_cal, y_cal)
    
    train_dataset = CovidDataset(train_df, train_transform, class_roots)
    cal_dataset = CovidDataset(cal_df, eval_transform, class_roots)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    cal_loader = DataLoader(cal_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Create and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_advanced_model(num_classes=3)
    model = model.to(device)
    
    # Train
    trained_model, train_losses, val_accuracies = train_model(
        model, train_loader, cal_loader, device, num_epochs=20)

if __name__ == "__main__":
    main()