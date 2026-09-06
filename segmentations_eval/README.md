# Semantic Segmentation Evaluation

Example script for training semantic segmentation models using pretrained DeepAndes backbone with a simple linear segmentation head. Also supports multiple baseline models including MAE, MoCoV2, SatMAE, and training from scratch.

## Overview

This repository contains code for evaluating semantic segmentation performance on 8-band satellite imagery. The training script performs 5-fold cross-validation and tracks metrics using Weights & Biases.

## Dataset Format

The dataset should be organized as follows:

```
dataset_folder/
├── images/
│   ├── image_1.npy
│   ├── image_2.npy
│   └── ...
└── masks/
    ├── image_1.npy
    ├── image_2.npy
    └── ...
```

- **Images**: NumPy arrays (`.npy` format) containing 8-band satellite imagery
- **Masks**: NumPy arrays (`.npy` format) containing binary segmentation masks
- **Example dataset** is provided in [**active_corrals_data**](./active_corrals_data/)


## Configuration Files

The [**configs**](./configs/) contains example YAML configuration files for different model backbones on **activate corral segmentation task**.

The `model_name` in config YAML file supports the following backbone options:

- `deepandes` — our ViT-L model from DINOv2
- `mae` — Masked Autoencoder
- `mocov2` — Momentum Contrast v2
- `satmae` — A Satellite MAE baseline
- `scratch` — randomly initialized ViT-L (no pre-training)

The MoCoV2 pre-trained weight (moco_v2_200ep_pretrain.pth.tar) can be downloaded from offical github [download here](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar).


## Usage

### Training

Train a model using a configuration file:

```bash
python main_binary_experiment.py --config /path/to/config.yaml
```

### Example: Active Corral Segmentation

Train using the DeepAndes (FM3M, pretrained 3 million) configuration:

```bash
python main_binary_experiment.py --config configs/corrals_active_FM3M.yaml
```


### Configuration Setup

Before training, edit the configuration YAML file to set the correct paths:

1. **Data paths**: Update `image_folder` and `mask_folder` to point to your dataset
2. **Output directory**: Set `output_dir` to where you want to save model checkpoints and visualizations
3. **Weights & Biases**: Update `project` and `name` for experiment tracking
4. **Pretrained weights**: If using a pretrained model, specify the path in `pretrained_weights` (if necessary)


## Training Details

- **Cross-validation**: 5-fold cross-validation is performed automatically
- **Evaluation metrics**: Dice Score and Mean IoU
- **Visualizations**: Predictions are saved every 25 epochs (if `output_dir` is specified)
- **Backbone freezing**: Set `train_backbone: False` to freeze the backbone and only train the segmentation head

## Outputs

When `output_dir` is specified, the script saves:
- Model checkpoints at epoch intervals
- Validation visualizations (RGB image, prediction overlay, ground truth)
- Prediction masks (NumPy format) for each validation sample

Output structure:
```
output_dir/
└── fold{fold_number}_{epoch}ep/
    ├── image_0.jpg
    ├── image_0.npy
    ├── image_1.jpg
    └── ...
```
