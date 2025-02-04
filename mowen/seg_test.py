import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from mowen_segmentation import MOWEN_Segmentation
from sklearn.metrics import f1_score, jaccard_score

# Define Segmentation Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),  # Resize to MOWEN's expected input size
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),  # Ensure masks match the output size
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        img = self.transform(img)
        mask = self.mask_transform(mask)
        
        mask = (mask > 0.5).float()  # Convert to binary 0/1 tensor
        
        return img, mask

# Load and Split Data
image_folder = 'scans-AG'
mask_folder = 'ground-truth-pixel-AG'
full_dataset = SegmentationDataset(image_folder, mask_folder)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

# Define Loss & Optimizer
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

criterion = lambda preds, targets: nn.BCEWithLogitsLoss()(preds, targets) + DiceLoss()(preds, targets)

# Load Model
pretrained_weights = "D:/mowen_epoch46.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MOWEN_Segmentation(pretrained_weights=pretrained_weights).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Training Loop
EPOCHS = 50
best_iou = 0.0
save_path = "mowen_segmentation_best.pth"
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        outputs = nn.functional.interpolate(outputs, size=(256, 256), mode="bilinear", align_corners=False)
        
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss, ious, f1s = 0.0, [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = nn.functional.interpolate(outputs, size=(256, 256), mode="bilinear", align_corners=False)
            
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            iou = jaccard_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='binary')
            f1 = f1_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='binary')
            
            ious.append(iou)
            f1s.append(f1)
    
    avg_iou = np.mean(ious)
    avg_f1 = np.mean(f1s)
    
    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(model.state_dict(), save_path)
        print(f"✅ Epoch {epoch+1}: New Best IoU: {best_iou:.4f}, F1: {avg_f1:.4f} - Model Saved!")
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f} - IoU: {avg_iou:.4f} - F1 Score: {avg_f1:.4f}")

end_time = time.time()
total_time = end_time - start_time
throughput = len(train_loader.dataset) * EPOCHS / total_time
print(f"🎉 Training Complete! Total Time: {total_time:.2f} sec - Throughput: {throughput:.2f} images/sec")

def evaluate_mowen(model, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0.0
    ious, f1s = [], []
    start_time = time.time()

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            
            # Resize outputs to match mask size
            outputs = nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            masks = (masks > 0.5).float()
            ious.append(jaccard_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='binary'))
            f1s.append(f1_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='binary'))
    
    throughput = len(test_loader.dataset) / (time.time() - start_time)
    mean_iou = np.mean(ious)
    mean_f1 = np.mean(f1s)

    print(f"Test Loss: {test_loss/len(test_loader):.4f}, Mean IoU: {mean_iou:.4f}, Mean F1: {mean_f1:.4f}, Throughput: {throughput:.2f} images/sec")
    return mean_iou, mean_f1, throughput

# Run Evaluation
mean_iou, mean_f1, throughput = evaluate_mowen(model, test_loader)
