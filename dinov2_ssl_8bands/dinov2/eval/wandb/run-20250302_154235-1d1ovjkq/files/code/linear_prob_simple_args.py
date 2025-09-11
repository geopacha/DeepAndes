# This code is customized to see the evaluation of finetuning classification layers on top of frozen patch embeddings
import sys

# replace the following path with the local address of the project folder 
# sys.path.append('/workspace/geopacha/dinov2_main_test') 
sys.path.append('/mnt/Data/guoj5/DGX_dump/geopacha/dinov2_main_test')


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
    timm_IM = args.use_timm

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


    if use_wandb:
        wandb.login(key="6e70d1ef3206d5f61cb24015681bf194979a8a33") # optional IN CLI: export WANDB_API_KEY=6e70d1ef3206d5f61cb24015681bf194979a8a33
          
        if wandb_project is not None: 
            print(f"WandB logging is enabled with project name: {wandb_project}")
        
        if wandb_trial is not None:
            print(f"WandB logging is enabled with project run: {wandb_trial}")
    
        run = wandb.init(project=wandb_project, name=wandb_trial)



    ############ simple loading ckpt ######################################## 
    if not timm_IM:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
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
        #change shape of pos_embed, shape depending on vits or vitg, or  vitl
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

    if timm_IM:
        print('load a plain model')
        model = timm.create_model(model_name='vit_large_patch16_224', pretrained=False, 
                                  in_chans=8, num_classes=2)


    model.head = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )

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

    # freeze optimizer
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
        os.path.join(output_dir, 'models', f'epoch_{epoch}_checkpoint.pth'))



    pd.DataFrame(logs, index=np.arange(1, epochs + 1)).to_csv(
        os.path.join(output_dir, 'logs.csv'), index_label='epoch')


if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument('--wandb_project', type=str, default="linear_prob_classic", help='Wandb project name')
    parser.add_argument('--wandb_trial', type=str, default="train_scratch_10%_ep10", help='Wandb trial name')
    parser.add_argument('--train_dataset_str', type=str, default="/workspace/data/junlin_classic/scaled/fold_1_10%/train", help='Training dataset path')
    parser.add_argument('--val_dataset_str', type=str, default="/workspace/data/junlin_classic/fold_1/val", help='Validation dataset path')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--output_dir', type=str, default="/workspace/geopacha/output_linear_prob_classic/scaled_10%/train_scratch_ep10", help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--pretrained_weights', type=str, default="/workspace/data/output/unsupervised/from_container/NEH_new/6gpus/eval/training_324999/teacher_checkpoint.pth", help='Pretrained weights path')
    
    # if not include in the CLI, they will be default to False. If add --flag in cli, they will be True
    parser.add_argument('--use_timm', action='store_true', help='Enable TIMM (default is False)')
    parser.add_argument('--use_wandb', action='store_true', help='(default is False)')
    args = parser.parse_args()
    
    main(args)

        # use_timm is False, use_wandb is True in the follow case: 
        # python /workspace/geopacha/dinov2_main_test/dinov2/eval/linear_prob_simple_args.py \
        # --use_wandb  \
        # --wandb_project "$WANDB_PROJECT" \
        # --wandb_trial "$WANDB_TRIAL" \
        # --train_dataset_str "$TRAIN_DATASET_STR" \
        # --val_dataset_str "$VAL_DATASET_STR" \
        # --cuda $CUDA \
        # --output_dir "$OUTPUT_DIR" \
        # --epochs $EPOCHS \
        # --pretrained_weights "$PRETRAINED_WEIGHTS" 