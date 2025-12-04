import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os
import requests
import zipfile
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import SimCLR

# Configuration
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "simclr_model_RN101.pth"  # Assumes file is in current directory
DATA_DIR = "./data"

def download_uc_merced(root):
    """
    Downloads and extracts the UC Merced dataset.
    """
    url = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"
    target_dir = os.path.join(root, "UCMerced_LandUse")
    
    # If the folder exists, we assume it's downloaded
    if os.path.exists(target_dir):
        print(f"Dataset folder found at {target_dir}")
        # Check if it contains Images folder
        images_dir = os.path.join(target_dir, "Images")
        if os.path.exists(images_dir):
             return images_dir
        # If not, maybe it's just the zip extracted directly
        return target_dir

    print(f"Downloading UC Merced dataset from {url}...")
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "UCMerced_LandUse.zip")
    
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(zip_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    
    # The zip usually extracts to UCMerced_LandUse/Images
    images_dir = os.path.join(root, "UCMerced_LandUse", "Images")
    if not os.path.exists(images_dir):
        # Fallback if structure is different
        return os.path.join(root, "UCMerced_LandUse")
        
    return images_dir

def get_uc_merced_loader(root, batch_size=64, split='train'):
    """
    Loads UC Merced dataset using ImageFolder.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Checking/Downloading dataset...")
    data_dir = download_uc_merced(root)
    print(f"Loading data from {data_dir}")
    
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except Exception as e:
        print(f"Error loading ImageFolder: {e}")
        return None, None

    # Create train/test split
    # Stratified split to ensure class balance
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42, stratify=dataset.targets)
    
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

def load_simclr_model(model_path, device):
    print(f"Loading model from {model_path}...")
    # Initialize backbone
    backbone = models.resnet101(weights=None) # We load our own weights
    
    # Initialize SimCLR wrapper
    model = SimCLR(backbone=backbone, tau=0.1)
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load state dict directly: {e}")
        print("Attempting to load as full model and extract state_dict...")
        try:
            full_model = torch.load(model_path, map_location=device)
            model.load_state_dict(full_model.state_dict())
        except Exception as e2:
            print(f"Failed to load full model: {e2}")
            raise e
            
    model.to(device)
    model.eval()
    return model

import torch.optim as optim
import time
import copy

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda'):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # SimCLR wrapper returns features in .encode(), but here we need logits.
                    # We need to add a classification head to the backbone.
                    # The backbone output is (B, 2048).
                    features = model.backbone(inputs)
                    outputs = model.classifier(features)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

class FineTuneModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super(FineTuneModel, self).__init__()
        self.backbone = backbone
        # ResNet101 fc input features
        num_ftrs = 2048 
        self.classifier = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    print("Preparing Data...")
    train_loader, test_loader = get_uc_merced_loader(DATA_DIR, BATCH_SIZE)
    if train_loader is None:
        return
    
    dataloaders = {'train': train_loader, 'val': test_loader}

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        return

    simclr_model = load_simclr_model(MODEL_PATH, DEVICE)
    
    # 3. Prepare for Fine-Tuning
    print("Setting up Fine-Tuning...")
    # We use the backbone from SimCLR
    backbone = simclr_model.backbone
    # Ensure backbone is trainable (unfrozen)
    for param in backbone.parameters():
        param.requires_grad = True
        
    # Create the full model with classification head
    # UC Merced has 21 classes
    num_classes = 21
    model = FineTuneModel(backbone, num_classes).to(DEVICE)
    
    # 4. Setup Training
    criterion = nn.CrossEntropyLoss()
    
    # Use a lower learning rate for the backbone, higher for the head? 
    # Or just a small LR for everything.
    optimizer = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-5}, # Low LR for pretrained layers
        {'params': model.classifier.parameters(), 'lr': 1e-3} # Higher LR for new head
    ])
    
    # 5. Train
    print("Starting Fine-Tuning (this will take longer than linear eval)...")
    model, hist = train_model(model, dataloaders, criterion, optimizer, num_epochs=15, device=DEVICE)
    
    # 6. Final Evaluation
    print("Final Evaluation on Test Set...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    main()
