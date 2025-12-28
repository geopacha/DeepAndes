"""
Semantic Segmentation 

Example for training a semantic segmentation model using pretrained DeepAndes backbone 
with a simple linear segmentation head. Also supports other baseline backbones including
MAE, MoCoV2, SatMAE, and Scratch (training from scratch).

Performs K-fold (K=5) cross-validation and tracks metrics using Weights & Biases.
"""

# Use python (/usr/bin/python), python 3.10.12
import torch
import torch.nn as nn

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob 
from tqdm import tqdm 
import os 

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm 


import wandb 
import timm 
import cv2 
import matplotlib.pyplot as plt 

from sklearn.model_selection import KFold

# import from train_utils 
from train_utils import  MaeForSemanticSegmentation_timm
from train_utils import Dinov2ForSemanticSegmentation, Dinov2ForSemanticSegmentation_timm
from train_utils import SegmentationDataset
from train_utils import dice_score, mean_iou


import yaml
import argparse

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)  # Load the YAML file content as a dictionary
    return config


parser = argparse.ArgumentParser(description="Parse the path to config yaml file from the command line.")
parser.add_argument('--config', type=str, required=True, help='path to train config yaml')
args = parser.parse_args()

train_config = load_config(args.config)


image_folder = train_config['image_folder']
mask_folder = train_config['mask_folder']
train_backbone = train_config['train_backbone']
train_from_scratch = train_config['train_from_scratch'] # equivalent to model_name = 'scratch'
learning_rate = float(train_config["learning_rate"])
epochs = train_config['epochs']
device = torch.device("cuda:0") 
batch_size = train_config['batch']
output_dir =  train_config.get('output_dir')
seed = train_config['seed']
model_name = train_config['model_name']  # pre-trained model name 
pretrained_weights = train_config.get('pretrained_weights', '')


# wandb login 
if train_config['wandb']:
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project=train_config['project'],
        name=train_config['name'],
        
        # Track hyperparameters and run metadata
        config=train_config
        )
    
# Get all image and mask file names
image_files = sorted(os.listdir(image_folder))
mask_files = [f for f in sorted(os.listdir(mask_folder)) if f.endswith('.npy')]  # Filter for .npy files only

# data transform 
resize_transform = A.Compose([
    A.Resize(224, 224),  # Target size (H, W)
    A.HorizontalFlip(p=0.5), 
    A.Rotate(limit=45, p=1.0, border_mode=0), 

    ToTensorV2()       # Converts to PyTorch tensor
])

test_transform = A.Compose([
    A.Resize(224, 224),  # Target size (H, W)
    ToTensorV2()       # Converts to PyTorch tensor
])


kf = KFold(n_splits=5, shuffle=True, random_state=seed) # K-fold cross validation (K=5)
fold_mean_dice = np.zeros(5)
for fold, (train_indices, val_indices) in enumerate(kf.split(image_files)):
 
    print(f"Fold {fold+1}")

    # Select the corresponding files for training and validation
    train_image_files = [image_files[i] for i in train_indices]
    val_image_files = [image_files[i] for i in val_indices]
    train_mask_files = [mask_files[i] for i in train_indices]
    val_mask_files = [mask_files[i] for i in val_indices]


     # Create datasets for this fold
    train_dataset = SegmentationDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
        transform=resize_transform,
        image_files=train_image_files,
        mask_files=train_mask_files
    )
    
    val_dataset = SegmentationDataset(
        image_folder=image_folder,
        mask_folder=mask_folder,
        transform=test_transform,
        image_files=val_image_files,
        mask_files=val_mask_files
    )

    # Create DataLoader for this fold
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if model_name == 'deepandes':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        teacher_checkpoint = pretrained_weights # path to .pth 
        pretrained_dict = torch.load(teacher_checkpoint)
        checkpoint_key = 'teacher'
        new_state_dict = {}
        for k, v in pretrained_dict[checkpoint_key].items():
            if 'dino_head' in k:
                print(f'{k} not used')

            elif 'ibot_head' in k:
                print(f'{k} not used')
            else:
                new_key = k.replace('backbone.', '')
                new_state_dict[new_key] = v

        pos_embed = nn.Parameter(torch.zeros(1, 257, 1024))
        model.pos_embed = pos_embed

        new_patch_embed = model.patch_embed
        new_patch_embed.proj = nn.Conv2d(
            in_channels=8,  # Updated for 8 input bands
            out_channels=new_patch_embed.proj.out_channels,
            kernel_size=new_patch_embed.proj.kernel_size,
            stride=new_patch_embed.proj.stride,
            padding=new_patch_embed.proj.padding,
            # bias=new_patch_embed.proj.bias,  
        )

        # Replace the old PatchEmbed with the updated one
        model.patch_embed = new_patch_embed
        # load state dict
        model.load_state_dict(new_state_dict, strict=True)

        segmentation_model = Dinov2ForSemanticSegmentation(backbone=model, hidden_size=1024, num_labels=1)

    if model_name == 'scratch' or train_from_scratch:
        print('\n train from scratch \n')
        model = timm.create_model(model_name='vit_large_patch14_224', pretrained=False, 
                                        in_chans=8)
        model.head = torch.nn.Identity()

        segmentation_model = Dinov2ForSemanticSegmentation_timm(backbone=model, hidden_size=1024, num_labels=1)  

    if model_name == 'mae':
        model = timm.create_model('vit_large_patch16_224.mae', pretrained=True,num_classes = 0, in_chans = 8)
        segmentation_model = MaeForSemanticSegmentation_timm(backbone=model, hidden_size=1024, num_labels=1)

    if model_name == 'mocov2':
        from torchvision.models import resnet50
        from other_baselines import load_moco_backbone
        from train_utils_moco import MocoV2ForSemanticSegmentation

        model = resnet50(pretrained=False)   # or True if ImageNet init 
        model.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.avgpool = nn.Identity()
        model.fc = nn.Identity() 

        moco_weights_path = pretrained_weights
        model = load_moco_backbone(model, moco_weights_path, in_chans=8)

        segmentation_model = MocoV2ForSemanticSegmentation(backbone=model, hidden_size=2048, num_labels=1)

    if model_name == 'satmae':
        from other_baselines import satmae_backbone_builder
        from train_utils import SatMAEForSemanticSegmentation

        # load satmae model (backbone)
        satmae_model = satmae_backbone_builder()
        satmae_model.adapt_patch_embed_in_chans(new_in_ch=8, mode="mean")
        model = satmae_model.model
        print(f"SatMAE model loaded")
        
        # add a segmentation head
        segmentation_model = SatMAEForSemanticSegmentation(backbone=model, hidden_size=1024, num_labels=1)


    ## Train backbone or the entire network. 
    if not train_backbone: 
        if model_name == 'deepandes':
            for name, param in segmentation_model.named_parameters():
                if name.startswith("dinov2"):
                    param.requires_grad = False 
        else:
            for name, param in segmentation_model.named_parameters():
                if name.startswith("backbone"):
                    param.requires_grad = False 
    
    optimizer = AdamW(segmentation_model.parameters(), lr=learning_rate)

    segmentation_model.to(device)
    segmentation_model.train()


    ## Train loop 
    max_dice = 0
    for epoch in tqdm(range(epochs)):
        print("Epoch:", epoch)
        
        segmentation_model.train()
        for idx, batch in enumerate(train_dataloader):
            pixel_values = batch[0].to(device)
            labels = batch[1].to(device)

            # forward pass
            logits, loss = segmentation_model(pixel_values, labels=labels.float())

            loss.backward()
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if idx % 20 == 0 and train_config['wandb']:
                wandb.log({"train_loss":loss.item()})


            
        segmentation_model.eval()
        total_dice = 0.0
        total_iou = 0.0
        total_samples = 0


        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                pixel_values = batch[0].to(device)
                labels = batch[1].to(device)

                # forward pass
                logits, _ = segmentation_model(pixel_values)

                # Compute Dice score and F1 score for the current batch
                dice = dice_score(logits, labels)
                iou = mean_iou(logits, labels, threshold=0.5)
            
                # Accumulate the metrics for averaging later
                total_dice += dice.item() * labels.shape[0]  # Multiply by batch size
                total_iou += iou.item() * labels.shape[0]
                total_samples += labels.shape[0]
            
            # Average the metrics over the whole test set
            avg_dice = total_dice / total_samples
            avg_iou = total_iou / total_samples

            print(f"Average Dice Score: {avg_dice:.4f}")
            print(f"Average IoU Score: {avg_iou:.4f}")

            if train_config['wandb']:
                wandb.log({"mean_dice": avg_dice, "mean_IoU": avg_iou, "epoch": epoch})   

            if max_dice < avg_dice:
                max_dice = avg_dice 

    ## visualization 
        if (epoch+1) % 25 == 0 and output_dir is not None:
            output_vis = os.path.join(output_dir, f'fold{fold}_{epoch+1}ep')
            os.makedirs(output_vis, exist_ok=True)
            threshold = 0.5
            for idx in range(len(val_dataset)):
                image, gt = val_dataset[idx]

                with torch.no_grad():
                    pixel_values = image.unsqueeze(0) # (b, c, h, w)
                    logits, _ = segmentation_model(pixel_values.to(device))



                pred = torch.sigmoid(logits.squeeze(1))  # Apply sigmoid to logits, (b, h, w)
                predicted_map = (pred > threshold).float()  # Threshold to get binary mask

                color_mask = np.zeros_like(image.permute(1,2,0).detach().cpu().numpy()[:,:,(4, 2, 1)])
                color_mask[predicted_map.squeeze(0).detach().cpu().numpy() > 0 ] = [0, 1., 0]  # Red color for the foreground
                alpha = 0.5  # Transparency factor

                rgb_image = image.permute(1,2,0).detach().cpu().numpy()[:,:,(4, 2, 1)]
                overlay = cv2.addWeighted(rgb_image, 1 - alpha, color_mask, alpha, 0)



                fig, axes = plt.subplots(1, 3, figsize=(6, 2))

                # Plot the original RGB image
                axes[0].imshow(rgb_image)
                axes[0].set_title("RGB bands")
                axes[0].axis('off')

                # Plot the overlay image
                axes[1].imshow(overlay)
                axes[1].set_title("Prediction")
                axes[1].axis('off')


                axes[2].imshow(gt.detach().cpu().numpy(), cmap='gray')
                axes[2].axis('off')
                axes[2].set_title("GT")

                # Display the plot
                plt.tight_layout()
                plt.savefig(os.path.join(output_vis, f'image_{idx}.jpg'))
                plt.close()


                #save prediction only
                np.save(os.path.join(output_vis,  f'image_{idx}.npy'), predicted_map.squeeze(0).detach().cpu().numpy())
                

    fold_mean_dice[int(fold)] = max_dice
    wandb.log({"fold_DSC": max_dice})
    wandb.log({"fold_mean_dice": fold_mean_dice.mean()})
    