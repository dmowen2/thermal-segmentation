import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from timm.models.vision_transformer import VisionTransformer

class CNNFeatureViT(VisionTransformer):
    def __init__(self, embed_dim=768, depth=12, num_heads=12):
        super().__init__(
            img_size=14,  # CNN already extracted 14x14 patches
            patch_size=1,  # No need for extra patching
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            num_classes=embed_dim
        )
    
    def forward(self, x):
        return self.forward_features(x)

class MOWEN_Segmentation(nn.Module):
    def __init__(self, img_size=256, embed_dim=768, depth=12, num_heads=12, pretrained_weights=None):
        super(MOWEN_Segmentation, self).__init__()
        
        # CNN Backbone (ResNet101)
        resnet = resnet101(weights='IMAGENET1K_V1')
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_projection = nn.Linear(2048, embed_dim)
        
        # ViT Backbone
        self.vit = CNNFeatureViT(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        
        # U-Net Style Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # Output single-channel mask
        )
        
        # Load Pretrained Weights
        if pretrained_weights:
            state_dict = torch.load(pretrained_weights, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print("✅ Loaded MAE Pretrained Weights!")
    
    def forward(self, x):
        cnn_features = self.cnn(x)  # (B, 2048, 14, 14)
        patches = self.prepare_patches(cnn_features)  # (B, 196, 768)
        encoded_patches = self.vit.blocks(patches)  # (B, 196, 768)
        
        # Reshape ViT output to 14x14 feature maps
        B = encoded_patches.shape[0]
        encoded_patches = encoded_patches.permute(0, 2, 1).contiguous().view(B, 768, 14, 14)
        
        # Decode to 256x256
        segmentation_output = self.decoder(encoded_patches)
        segmentation_output = F.interpolate(segmentation_output, size=(256, 256), mode='bilinear', align_corners=False)
        return segmentation_output

    def prepare_patches(self, features):
        B, C, H, W = features.shape
        patches = features.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        patches = self.feature_projection(patches)
        return patches

#D:/mowen_epoch46.pth