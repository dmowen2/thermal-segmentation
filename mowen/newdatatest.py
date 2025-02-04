import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score
import torch.nn as nn
from mowen_segmentation import MOWEN_Segmentation

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
best_weights = "mowen_segmentation_best.pth"
model = MOWEN_Segmentation(pretrained_weights=best_weights).to(device)
model.eval()
print("✅ Loaded Best MOWEN Weights for Testing!")

# Preprocess Image
def preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # Add batch dim
    return img

# Fix model output shape issue
def reshape_mowen_output(output, target_size=(256, 256)):
    if output.dim() == 3:  # Expected (B, N, 768)
        B, N, C = output.shape
        H, W = int(np.sqrt(N)), int(np.sqrt(N))  # Should be 14x14
        output = output.permute(0, 2, 1).contiguous().view(B, C, H, W)  # Reshape to (B, 768, 14, 14)
    elif output.dim() == 4:  # Expected (B, C, H, W)
        B, C, H, W = output.shape
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")
    
    output = nn.functional.interpolate(output, size=target_size, mode='bilinear', align_corners=False)
    return output

# Visualize Predictions
def visualize_predictions(image_paths, pred_paths, ground_truth_paths):
    for i in range(len(image_paths)):
        img = cv2.imread(image_paths[i])
        pred_mask = cv2.imread(pred_paths[i], cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.imread(ground_truth_paths[i], cv2.IMREAD_GRAYSCALE)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(true_mask, cmap="gray")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_mask, cmap="gray")
        plt.axis("off")
        
        plt.show()

# Test Model and Save Results
def test_mowen(model, image_paths, ground_truth_paths=None, img_size=(224, 224), save_folder='mowen_output'):
    model.to(device)
    model.eval()
    os.makedirs(save_folder, exist_ok=True)
    csv_path = os.path.join(save_folder, 'metrics.csv')

    pred_paths = []
    ious, f1s = [], []
    with open(csv_path, 'w') as f:
        f.write("Image,IoU,F1-Score\n")  # CSV Header

    for i, image_path in enumerate(image_paths):
        img = preprocess_image(image_path, img_size)

        # Run inference
        with torch.no_grad():
            pred_mask = model(img)
            pred_mask = reshape_mowen_output(pred_mask, target_size=(256, 256))
            pred_mask = nn.functional.interpolate(
                pred_mask, size=(256, 256), mode='bilinear', align_corners=False
            ).squeeze().cpu().numpy()
            print(f"Raw Prediction - Min: {pred_mask.min()}, Max: {pred_mask.max()}, Mean: {pred_mask.mean()}")

            pred_mask = torch.sigmoid(torch.tensor(pred_mask))  # Normalize to 0-1
            pred_mask_binary = (pred_mask > 0.2).numpy().astype(np.uint8)


        # Save predicted mask
        pred_mask_path = os.path.join(save_folder, f'pred_mask_{i:02d}.png')
        cv2.imwrite(pred_mask_path, pred_mask_binary * 255)
        pred_paths.append(pred_mask_path)

        # Compute IoU and F1-score if ground truth exists
        if ground_truth_paths:
            true_mask = cv2.imread(ground_truth_paths[i], cv2.IMREAD_GRAYSCALE)
            true_mask = cv2.resize(true_mask, (256, 256))
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
        visualize_predictions(image_paths, pred_paths, ground_truth_paths)

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
test_mowen(model, new_image_paths, ground_truth_paths, save_folder='mowen_output')
