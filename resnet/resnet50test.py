import os
import torch
import numpy as np
import cv2
from sklearn.metrics import f1_score, jaccard_score
import timm
import torch.nn as nn

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the U-Net model with VGG19 encoder
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        # Use VGG19 as the encoder
        self.encoder = timm.create_model('resnet50', pretrained=True, features_only=True, out_indices=(0, 1, 2, 3, 4))
        
        # Extract the feature extraction layers from InceptionV3
        self.encoder_layers = lambda x: self.encoder(x)[-1]
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
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

# Preprocess a new image
def preprocess_image(image_path, img_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    return img

# Test the model on new images and save results
def test_new_images(model, image_paths, ground_truth_paths=None, img_size=(256, 256), save_folder='vggimg/vgg19'):
    model.to(device)
    model.eval()

    os.makedirs(save_folder, exist_ok=True)
    csv_path = os.path.join(save_folder, 'metrics.csv')

    ious = []
    f1s = []

    with open(csv_path, 'w') as f:
        f.write("Image,IoU,F1-Score\n")  # CSV Header

    for i, image_path in enumerate(image_paths):
        img = preprocess_image(image_path, img_size).to(device)

        # Run inference
        with torch.no_grad():
            pred_mask = model(img)
            pred_mask = nn.functional.interpolate(
                pred_mask, size=img_size, mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()
            pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)

        # Save predicted mask
        pred_mask_path = os.path.join(save_folder, f'pred_mask_{i:02d}.png')
        cv2.imwrite(pred_mask_path, pred_mask_binary * 255)

        # Compute IoU and F1-score if ground truth exists
        if ground_truth_paths:
            true_mask = cv2.imread(ground_truth_paths[i], cv2.IMREAD_GRAYSCALE)
            true_mask = cv2.resize(true_mask, img_size)
            true_mask = (true_mask / 255.0).astype(np.uint8)

            iou = jaccard_score(true_mask.flatten(), pred_mask_binary.flatten(), average='binary')
            f1 = f1_score(true_mask.flatten(), pred_mask_binary.flatten(), average='binary')
            ious.append(iou)
            f1s.append(f1)

            # Append results to CSV
            with open(csv_path, 'a') as f:
                f.write(f"{image_path},{iou:.4f},{f1:.4f}\n")

            print(f"Saved: {pred_mask_path} | IoU: {iou:.4f}, F1-score: {f1:.4f}")

    if ground_truth_paths:
        print(f"\nMean IoU: {np.mean(ious):.4f}")
        print(f"Mean F1-score: {np.mean(f1s):.4f}")

# Load the saved model weights
weights_path = "best_weights_resnet50.pth"  # Ensure you have the correct weights file
model = UNet(num_classes=1).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
print("Model weights loaded successfully.")

# Paths to new images and their ground truth masks
new_image_paths = [
    r"C:\Users\bluem\ANLResearch\scans\scans\00.png",
    r"C:\Users\bluem\ANLResearch\scans\scans\01.png",
    r"C:\Users\bluem\ANLResearch\scans\scans\07.png",
    r"C:\Users\bluem\ANLResearch\scans\scans\05.png",
    r"C:\Users\bluem\ANLResearch\scans\scans\14.png",
    r"C:\Users\bluem\ANLResearch\scans\scans\15.png"
]
ground_truth_paths = [
    r"C:\Users\bluem\ANLResearch\ground-truth-pixel\ground-truth-pixel\00.png",
    r"C:\Users\bluem\ANLResearch\ground-truth-pixel\ground-truth-pixel\01.png",
    r"C:\Users\bluem\ANLResearch\ground-truth-pixel\ground-truth-pixel\07.png",
    r"C:\Users\bluem\ANLResearch\ground-truth-pixel\ground-truth-pixel\05.png",
    r"C:\Users\bluem\ANLResearch\ground-truth-pixel\ground-truth-pixel\14.png",
    r"C:\Users\bluem\ANLResearch\ground-truth-pixel\ground-truth-pixel\15.png"
]

# Test the model and save predicted masks & CSV metrics
test_new_images(model, new_image_paths, ground_truth_paths, save_folder='resnetimg/resnet50')
