# This code is customized to see the evaluation of finetuning classification on top of different backbones
import sys

# replace the following path with the local address of the project folder 
sys.path.append('/path/to/dinov2_ssl_8bands')


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

## add
from dinov2.data.datasets import LBDataset
import albumentations as A
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

import wandb 
import os 
import timm 

import argparse


def main(args):

     # Assign arguments to variables
    use_wandb = args.use_wandb
    wandb_project = args.wandb_project
    wandb_trial = args.wandb_trial
    train_dataset_str = args.train_dataset_str
    val_dataset_str = args.val_dataset_str
    cuda = args.cuda
    output_dir = args.output_dir
    epochs = args.epochs
    pretrained_weights = args.pretrained_weights
    timm_IM = args.timm_IM
    model_name = args.model_name

    # Now you can use these variables in your training code
    print("\n\nVariables\n")
    print(f"Using wandb: {use_wandb}")
    print(f"Wandb project: {wandb_project}")
    print(f"Wandb trial: {wandb_trial}")
    print(f"Train dataset path: {train_dataset_str}")
    print(f"Validation dataset path: {val_dataset_str}")
    print(f"CUDA device: {cuda}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Pretrained weights: {pretrained_weights}")
    print(f"timm_IM {timm_IM}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

    if use_wandb:
        wandb.login(key="your_wandb_api_key_paste_here") # optional IN CLI: export 'your_wandb_api_key'
          
        if wandb_project is not None: 
            print(f"WandB logging is enabled with project name: {wandb_project}")
        
        if wandb_trial is not None:
            print(f"WandB logging is enabled with project run: {wandb_trial}")
    
        run = wandb.init(project=wandb_project, name=wandb_trial)



    ############ simple loading ckpt ######################################## 
    if model_name =='deepandes':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')  # vitl14 is the default backbone for deepandes
        teacher_checkpoint = pretrained_weights # path to .pth 
        pretrained_dict = torch.load(teacher_checkpoint, map_location="cpu")
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
        # change shape of pos_embed, shape depending on vits or vitg, or  vitl, for deepandes (vitl14) and input of 224x224, 8 bands
        # there will be 256 patch tokens and 1 class tokens, so 257 tokens in total, each token has 1024 hidden dimensions
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

        model.load_state_dict(new_state_dict, strict=True)
        model.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    # Evaluate on MoCo-v2 backbone. expand the input channels to 8 bands
    if model_name =='mocov2':
        from torchvision.models import resnet50
        from dinov2.eval.other_baselines import load_moco_backbone
        
        # make a ResNet-50 for 8 channels
        model = resnet50(pretrained=False)   # or True if ImageNet init 
        model.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Identity() 


        # load MoCo weights to the plain model backbone
        moco_weights_path = pretrained_weights
        model = load_moco_backbone(model, moco_weights_path, in_chans=8)

        # add a head for classification 
        model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    # Evaluate on MAE backbone. expand the input channels to 8 bands
    if model_name =='mae':
        model = timm.create_model(
            'vit_large_patch16_224.mae',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
            in_chans=8,
        )
        model.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    # Evaluate on SATMAE backbone. expand the input channels to 8 bands
    if model_name =='satmae':
        from dinov2.eval.other_baselines import mae_backbone_builder, mae_classifier
        satmae_model = mae_backbone_builder()
        satmae_model.adapt_patch_embed_in_chans(new_in_ch=8, mode="mean")
        satmae_classifier = mae_classifier(satmae_model.model, num_classes=2, hidden_dim=256, freeze_decoder=True, base_embed_dim=1024)
        model = satmae_classifier


    # Evaluate on a Vit-L14 backbone. expand the input channels to 8 bands, trained from sratch
    if timm_IM:
        model = timm.create_model(model_name='vit_large_patch14_224', pretrained=False, 
                                  in_chans=8, num_classes=2)

        model.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    ####### Optional: freeze/unfreeze the model backbone######
    # for param in model.parameters():
    #     param.requires_grad = True

    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    train_transforms = A.Compose([
        A.Resize(height=224, width=224), # test (518, 518) for lvd 
        A.Flip(),
        A.RandomRotate90(), 
        ]
    )

    seed = 0
    log_minibatch = 100


    train_dataset = LBDataset(root=train_dataset_str, transforms=train_transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True
    )

    val_dataset = LBDataset(root=val_dataset_str, transforms=train_transforms)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=False)


    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters()}
        ], lr=1e-5)

    criterion = nn.CrossEntropyLoss()

    # Move the model to GPU if available
    cuda_device = cuda
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # logs
    logs = {'auc_score': [],
            'sensitivity': [],
            'specificity': [],
            'f1-score': [],
            'precision': []
            }

    for epoch in tqdm(range(epochs)):

        model.train()
        running_loss = 0.0

        ## Train Acc
        train_pred = []
        train_label = []
        running_corrects = 0
        total = 0


        for i, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % log_minibatch == log_minibatch - 1:  # print every log_mini-batches
                print('[%d, %5d] running loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / log_minibatch))
                if use_wandb:
                    wandb.log({"running_loss": round(running_loss / log_minibatch, 4)})
                running_loss = 0.0

            ## add train log
            outputs = torch.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)
            train_label.extend(labels.cpu().numpy())
            train_pred.extend(preds.cpu().detach().numpy())
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        ## training accuracy
        train_accuracy = running_corrects.double() / total
        print(f"\nEpoch {epoch + 1} Train Accuracy : {train_accuracy:.4f}")

        # Validation
        model.eval()
        running_corrects = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                ##### add by junlin ############
                outputs = torch.softmax(outputs, 1)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().detach().numpy())

        val_accuracy = running_corrects.double() / total
        print(f"\nEpoch {epoch + 1} Validation Accuracy : {val_accuracy:.4f}")

        auc_score = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1} Validation auc_score : {auc_score}\n")

        # conf_matrix = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = f1_score(all_labels, all_preds)
        print(f'f1 score ', 2*tp/(2*tp + fp + fn))
        ConfusionMatrixDisplay.from_predictions(y_true=all_labels, y_pred=all_preds)
        plt.title(f'Epoch {epoch + 1} conf_matrix')

        plt.savefig(os.path.join(output_dir, f'Epoch_{epoch + 1}_confMatrix.png'))
        plt.close()

        print(f"Epoch {epoch + 1} Validation Sensitivity : {sensitivity}\n")
        print(f"Epoch {epoch + 1} Validation Specificity : {specificity}\n")
        print(f"Epoch {epoch + 1} Validation Precision : {precision}\n")
        print(f"Epoch {epoch + 1} Validation F1 : {f1}\n")

        # save metrics
        logs['auc_score'].append(round(auc_score, 4))
        logs['sensitivity'].append(round(sensitivity, 4))
        logs['specificity'].append(round(specificity, 4))
        logs['f1-score'].append(round(f1, 4))
        logs['precision'].append(round(precision, 4))

 
        # log validation metrics
        if use_wandb:
            wandb.log({
                "auc_score": round(auc_score, 4),
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "precision": round(precision, 4),
                "f1":round(f1, 2)
            })

        
        # save chkpt 
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),}, 
        os.path.join(output_dir, 'models', f'epoch_{epoch+1}_checkpoint.pth'))



    pd.DataFrame(logs, index=np.arange(1, epochs + 1)).to_csv(
        os.path.join(output_dir, 'logs.csv'), index_label='epoch')


if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument('--wandb_project', type=str, default="linear_prob_classic", help='Wandb project name')
    parser.add_argument('--wandb_trial', type=str, default="train_run_1", help='Wandb trial name')
    parser.add_argument('--train_dataset_str', type=str, default="/path/to/classification/train/dataset/folder", help='Training dataset path')
    parser.add_argument('--val_dataset_str', type=str, default="/path/to/classification/val/dataset/folder", help='Validation dataset path')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--output_dir', type=str, default="/path/to/output_dir", help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--pretrained_weights', type=str, default="/path/to/pretraining/output_dir/eval/training_324999/teacher_checkpoint.pth", help='Pretrained weights path')
    parser.add_argument('--model_name', type=str, default='deepandes', help='model name')

    # if not include in the CLI, they will be default to False. If add --flag in cli, they will be True
    parser.add_argument('--timm_IM', action='store_true', help='(default is False), for evaluation of the Scratch model') # for evaluation of the Scratch model
    parser.add_argument('--use_wandb', action='store_true', help='(default is False)') # for logging the metrics to wandb

    args = parser.parse_args()
    
    main(args)