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


class LinearClassifier(torch.nn.Module):
        def __init__(self, in_channels, tokenW=16, tokenH=16, num_labels=1):
            super(LinearClassifier, self).__init__()

            self.in_channels = in_channels
            self.width = tokenW
            self.height = tokenH
            self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))


        def forward(self, embeddings):
            embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
            embeddings = embeddings.permute(0,3,1,2)

            return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(torch.nn.Module):
    def __init__(self, backbone, hidden_size, num_labels, loss_fn=CombinedLoss(bce_weight=1.0, dice_weight=10.)):
        super(Dinov2ForSemanticSegmentation, self).__init__()

        self.dinov2 = backbone
        self.classifier = LinearClassifier(hidden_size, 16, 16, num_labels)
        self.loss_fn = loss_fn


    def forward(self, pixel_values, labels=None):
        # use frozen features
        backbone = self.dinov2
        outputs = backbone.forward_features(pixel_values, masks=None)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs['x_norm_patchtokens']

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)

        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.squeeze(), labels.squeeze())

        return logits, loss
    

class Dinov2ForSemanticSegmentation_timm(torch.nn.Module):
    def __init__(self, backbone, hidden_size, num_labels, loss_fn=CombinedLoss(bce_weight=1.0, dice_weight=10.0)):
        super(Dinov2ForSemanticSegmentation_timm, self).__init__()

        self.dinov2 = backbone
        self.classifier = LinearClassifier(hidden_size, 16, 16, num_labels)
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = loss_fn

    def forward(self, pixel_values, labels=None):
        # use frozen features
        backbone = self.dinov2
        outputs = backbone.forward_features(pixel_values)

        
        # patch_embeddings = outputs['x_norm_patchtokens']
        patch_embeddings = outputs[:,1:]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:

            loss = self.loss_fn(logits.squeeze(), labels.squeeze())

        return logits, loss


class MaeForSemanticSegmentation_timm(torch.nn.Module):
    def __init__(self, backbone, hidden_size, num_labels, loss_fn=CombinedLoss(bce_weight=1.0, dice_weight=10.0)):
        super(MaeForSemanticSegmentation_timm, self).__init__()

        self.backbone = backbone
        self.classifier = LinearClassifier(hidden_size, 14, 14, num_labels)
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = loss_fn

    def forward(self, pixel_values, labels=None):
        # use frozen features
        backbone = self.backbone
        outputs = backbone.forward_features(pixel_values)

        
        # patch_embeddings = outputs['x_norm_patchtokens']
        patch_embeddings = outputs[:,1:]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:

            loss = self.loss_fn(logits.squeeze(), labels.squeeze())

        return logits, loss


class SatMAEForSemanticSegmentation(torch.nn.Module):
    def __init__(self, backbone, hidden_size, num_labels, loss_fn=CombinedLoss(bce_weight=1.0, dice_weight=10.0)):
        super(SatMAEForSemanticSegmentation, self).__init__()

        self.backbone = backbone
        self.classifier = LinearClassifier(hidden_size, 14, 14, num_labels)
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = loss_fn

    def forward(self, pixel_values, labels=None):
        # use frozen features
        backbone = self.backbone
        outputs = backbone.forward_encoder(pixel_values, mask_ratio=0.0)[0]
    
        patch_embeddings = outputs[:,1:]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:

            loss = self.loss_fn(logits.squeeze(), labels.squeeze())

        return logits, loss



class SegmentationDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None, image_files=None, mask_files=None):
        """
        Initialize the dataset.
        
        :param image_folder: Path to the folder containing the image .npy files.
        :param mask_folder: Path to the folder containing the mask .npy files.
        :param transform: Optional transformation to apply to the images.
        :param image_files: List of image file names for this fold.
        :param mask_files: List of mask file names for this fold.
        """
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_files = image_files if image_files is not None else sorted(os.listdir(image_folder))
        self.mask_files = mask_files if mask_files is not None else sorted(os.listdir(mask_folder))
        self.transform = transform

        assert len(self.image_files) == len(self.mask_files), "Mismatch in number of images and masks!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])

        image = np.load(image_path)
        mask = np.load(mask_path).astype(np.int64)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to torch tensor
        if image.dtype == torch.uint8:
            image = image.float() / 255.0  # Normalize the image to (0, 1)

        return image, mask.long()
    

def dice_score(pred, target, smooth=1e-6, threshold=0.5):
    """
    Compute the Dice score for binary segmentation.

    Args:
        pred (Tensor): Predicted logits of shape (batch_size, height, width).
        target (Tensor): Ground truth binary mask of shape (batch_size, height, width).
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        Tensor: Dice score value.
    """

    if pred.dim() == 4: 
        pred = pred.squeeze(1)
    
    if target.dim() == 4:
        target = target.squeeze(1)


    pred = torch.sigmoid(pred)  # Apply sigmoid to logits
    pred = (pred > threshold).float()  # Threshold to get binary mask

    intersection = (pred * target).sum(dim=(1, 2))  # Sum intersection across height and width
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))  # Sum union across height and width
    dice = (2. * intersection + smooth) / (union + smooth)  # Dice coefficient

    return dice.mean()  # Average Dice score over the batch


def mean_iou(pred, target, threshold=0.5):
    """
    Compute the mean Intersection over Union (IoU) for binary segmentation.

    Args:
        pred (Tensor): Predicted logits of shape (batch_size, height, width).
        target (Tensor): Ground truth binary mask of shape (batch_size, height, width).
        threshold (float): Threshold to convert predicted logits to binary mask.

    Returns:
        Tensor: Mean IoU value.
    """

    if pred.dim() == 4: 
        pred = pred.squeeze(1)
    
    if target.dim() == 4:
        target = target.squeeze(1)

    pred = torch.sigmoid(pred)  # Apply sigmoid to logits
    pred = (pred > threshold).float()  # Threshold to get binary mask

    intersection = (pred * target).sum(dim=(1, 2))  # Sum intersection across height and width
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection  # Sum union across height and width
    iou = (intersection + 1e-6) / (union + 1e-6)  # Compute IoU with small epsilon for stability

    return iou.mean()  # Return the mean IoU over the batch


