import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob 
from tqdm import tqdm 
import os 

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


import wandb 
import timm 
import cv2 
import matplotlib.pyplot as plt 

from sklearn.model_selection import KFold


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=1., dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        
        if logits.dim() == 3: 
            logits = logits.unsqueeze(1)
        
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # BCE Loss
        bce = self.bce_loss(logits, targets)

        # Dice Loss
        smooth = 1e-6  # Avoid division by zero
        pred = torch.sigmoid(logits)  # Convert logits to probabilities
        intersection = (pred * targets).sum(dim=(2, 3))  # Sum intersection across spatial dimensions
        union = pred.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))  # Sum union across spatial dimensions
        dice = (2. * intersection + smooth) / (union + smooth)  # Dice coefficient

        # Average Dice Loss across the batch
        dice_loss = 1 - dice.mean()  # Mean Dice Loss across the batch

        # Combined Loss (weighted sum of BCE and Dice Losses)
        combined_loss = self.bce_weight * bce + self.dice_weight * dice_loss
        return combined_loss



# ResNet Classifier
class ResNetClassifier(torch.nn.Module):
    def __init__(self, in_channels, num_labels=1):
        super(ResNetClassifier, self).__init__()
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, kernel_size=1)
    
    def forward(self, embeddings):
        # embeddings is already [batch_size, channels, height, width] for ResNet
        return self.classifier(embeddings)



# Modified MocoV2 class that handles reshape automatically
class MocoV2ForSemanticSegmentation(torch.nn.Module):
    def __init__(self, backbone, hidden_size, num_labels, loss_fn=CombinedLoss(bce_weight=1.0, dice_weight=10.0)):
        super(MocoV2ForSemanticSegmentation, self).__init__()
        
        self.backbone = backbone
        self.classifier = ResNetClassifier(hidden_size, num_labels)
        self.loss_fn = loss_fn
        self.hidden_size = hidden_size
    
    def forward(self, pixel_values, labels=None):
        # Get features from backbone
        features = self.backbone(pixel_values)
        
        # Handle both flattened and spatial features
        if features.dim() == 2:  # Flattened features [batch_size, 100352]
            # Reshape to spatial dimensions [batch_size, 2048, 7, 7]
            batch_size = features.size(0)
            features = features.view(batch_size, self.hidden_size, 7, 7)
        elif features.dim() == 4:  # Already spatial features [batch_size, 2048, 7, 7]
            pass  # No reshaping needed
        else:
            raise ValueError(f"Unexpected feature dimensions: {features.shape}")
        
        # Apply segmentation head
        logits = self.classifier(features)  # Shape: [batch_size, num_labels, 7, 7]
        
        # Upsample to input size
        logits = torch.nn.functional.interpolate(
            logits, 
            size=pixel_values.shape[2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.squeeze(), labels.squeeze())
        
        return logits, loss