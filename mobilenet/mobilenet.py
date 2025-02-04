import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, jaccard_score
import timm

# Define dataset class
class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        
        # Ensure both images and masks are resized to the same size
        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img, mask

# Load images and masks
def load_data(image_folder, mask_folder, img_size=(256, 256)):
    images = []
    masks = []
    for img_name in sorted(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_name)
        mask_path = os.path.join(mask_folder, img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize to [0, 1]

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = mask / 255.0

        images.append(img)
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    return images, masks

# Load dataset
image_folder = 'scans-AG'
mask_folder = 'ground-truth-pixel-AG'
images, masks = load_data(image_folder, mask_folder)
print("Image shape:", images.shape)
print("Mask shape:", masks.shape)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create datasets and dataloaders
train_dataset = SegmentationDataset(X_train, y_train)
val_dataset = SegmentationDataset(X_val, y_val)
test_dataset = SegmentationDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define U-Net model with DenseNet201 encoder
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
 # Use seresnext101 as the encoder
        self.encoder = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True, out_indices=(0, 1, 2, 3, 4))
        
        # Extract the feature extraction layers from InceptionV3
        self.encoder_layers = lambda x: self.encoder(x)[-1]
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(960, 256, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        features = self.encoder_layers(x)
        x = self.decoder(features)
        return x

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(num_classes=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Metrics
best_iou = 0
best_f1 = 0
lowest_train_loss = float('inf')
lowest_val_loss = float('inf')

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    global best_iou, best_f1, lowest_train_loss, lowest_val_loss
    total_start_time = time.time()
    weights_path = "best_weights_mobilenet.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)

            # Resize outputs to match mask size if necessary
            outputs = nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        ious = []
        f1s = []
        model.eval()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)

                # Resize outputs to match mask size
                outputs = nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                masks = (masks > 0.5).float()  # Convert masks to binary
                ious.append(jaccard_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='binary'))
                f1s.append(f1_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='binary'))

        avg_iou = np.mean(ious)
        avg_f1 = np.mean(f1s)

        # Update best metrics
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), weights_path)
            print(f"Epoch {epoch+1}: Saved best weights with IoU: {best_iou:.4f}")
        if avg_f1 > best_f1:
            best_f1 = avg_f1
        if train_loss / len(train_loader) < lowest_train_loss:
            lowest_train_loss = train_loss / len(train_loader)
        if val_loss / len(val_loader) < lowest_val_loss:
            lowest_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, IoU: {avg_iou:.4f}, F1: {avg_f1:.4f}")

    total_time = time.time() - total_start_time
    print(f"Total training time: {total_time / 60:.2f} minutes")
    return total_time

# Train the model
total_training_time = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50)

# Evaluate on test set
def evaluate_model(model, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0.0
    ious = []
    f1s = []
    start_time = time.time()

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            # Resize outputs to match mask size
            outputs = nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            
            loss = criterion(outputs, masks)
            test_loss += loss.item()

            preds = (outputs > 0.5).float()
            masks = (masks > 0.5).float()  # Convert masks to binary
            ious.append(jaccard_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='binary'))
            f1s.append(f1_score(masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='binary'))

    throughput = len(test_loader.dataset) / (time.time() - start_time)
    mean_iou = np.mean(ious)
    mean_f1 = np.mean(f1s)

    print(f"Test Loss: {test_loss/len(test_loader):.4f}, Mean IoU: {mean_iou:.4f}, Mean F1: {mean_f1:.4f}, Throughput: {throughput:.2f} images/sec")
    return mean_iou, mean_f1, throughput

mean_iou, mean_f1, throughput = evaluate_model(model, test_loader)

# Visualize predictions
def visualize_predictions(model, dataset, indices):
    model.to(device)
    model.eval()

    for idx in indices:
        img, true_mask = dataset[idx]
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask = model(img)

            # Resize prediction to match mask size
            pred_mask = nn.functional.interpolate(pred_mask, size=true_mask.shape[1:], mode='bilinear', align_corners=False).squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img.squeeze().permute(1, 2, 0).cpu().numpy())
        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(true_mask.squeeze(), cmap='gray')
        plt.subplot(1, 3, 3) 
        plt.title("Predicted Mask")
        plt.imshow(pred_mask, cmap='gray')
        plt.show()
