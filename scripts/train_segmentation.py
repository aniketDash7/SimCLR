import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms import functional as F
import os
import numpy as np
from PIL import Image
import requests
import zipfile
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SimCLR

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def download_sample_dataset(root):
    """
    Downloads Penn-Fudan Database for Pedestrian Detection and Segmentation
    as a sample dataset to verify the pipeline.
    User should replace this with WHU Building Dataset or similar for Remote Sensing.
    """
    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    target_dir = os.path.join(root, "PennFudanPed")
    
    if os.path.exists(target_dir):
        print("Sample dataset already exists.")
        return target_dir
        
    print(f"Downloading sample dataset from {url}...")
    zip_path = os.path.join(root, "PennFudanPed.zip")
    r = requests.get(url, stream=True)
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    
    return target_dir

class RemoteSensingDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # Setup for PennFudan structure. Modify for WHU/Standard format.
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        # Mask is usually 1-channel, where each pixel value is the instance ID
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        # Instance IDs
        obj_ids = np.unique(mask)
        # First ID is background, remove it
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64) # Single class (e.g., Building)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = F.to_tensor(img)
            # Add other transforms here if needed

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes, pretrained_simclr_path=None):
    # 1. Create ResNet-101 FPN backbone
    # trainable_layers=3 means we train the top 3 blocks.
    backbone = resnet_fpn_backbone('resnet101', weights=None, trainable_layers=3)
    
    # 2. Load SimCLR Weights
    if pretrained_simclr_path and os.path.exists(pretrained_simclr_path):
        print(f"Loading SimCLR weights from {pretrained_simclr_path} into Mask R-CNN backbone...")
        try:
            # SimCLR state dict keys: backbone.conv1.weight, backbone.layer1.0.conv1.weight...
            # FPN state dict keys: body.conv1.weight, body.layer1.0.conv1.weight...
            state_dict = torch.load(pretrained_simclr_path, map_location='cpu')
            
            # If saved as full model/wrapper
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    # Replace 'backbone.' with 'body.' to match FPN structure
                    new_key = k.replace('backbone.', 'body.')
                    # FPN backbone doesn't have 'fc' layer, so ignore it
                    if 'fc' not in new_key:
                        new_state_dict[new_key] = v
            
            # Load 'body' weights
            # strict=False because FPN has extra layers (fpn_inner, fpn_layer) that are not in ResNet
            missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded. Missing (expected for FPN layers): {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            print(f"Failed to load SimCLR weights: {e}")
            print("Using random initialization for backbone.")

    # 3. Create Mask R-CNN
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model

def main():
    # Paths
    DATA_DIR = "./data"
    SIMCLR_PATH = "simclr_model_RN101.pth"
    
    # 1. Prepare Data
    data_path = download_sample_dataset(DATA_DIR)
    dataset = RemoteSensingDataset(data_path)
    
    # Split
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset, indices[-50:])
    
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True,  num_workers=0, collate_fn=lambda x: tuple(zip(*x))
    )
    
    # 2. Setup Model
    # num_classes = 2 (background + building/pedestrian)
    model = get_model_instance_segmentation(num_classes=2, pretrained_simclr_path=SIMCLR_PATH)
    model.to(DEVICE)
    
    # 3. Optimization
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 4. Training Loop (Simple)
    num_epochs = 2 # Demo
    print("Starting Training...")
    
    for epoch in range(num_epochs):
        model.train()
        i = 0
        for images, targets in tqdm(data_loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss: {losses.item():.4f}")
            i += 1
            
        lr_scheduler.step()
        print(f"Epoch {epoch} complete.")
        
    # Save
    torch.save(model.state_dict(), "maskrcnn_simclr_tuned.pth")
    print("Model saved to maskrcnn_simclr_tuned.pth")

if __name__ == "__main__":
    main()
