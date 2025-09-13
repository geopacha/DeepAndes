# Multi-band Imagery Support for DINOv2
The DINOv2 framework was originally designed for RGB images (3 bands). However, in fields like remote sensing and earth observation, we often need **multispectral** or **hyperspectral** imagery (>3 bands).

## Purpose

- Extends Meta's DINOv2 (and potentially DINOv3) to support multispectral imagery pre-training (e.g., 8-band satellite imagery) 
- Maintain modularity while minimizing changes.



## Key Modifications

### 1. Dataset Module
**Location:** [`dinov2_ssl_8bands/dinov2/data/datasets/`](../dinov2_ssl_8bands/dinov2/data/datasets/)

- Added `nlb_dataset.py` module (extend the `extended.py`).
- Images are stored as individual `.npy` files, saved in a single folder.
- File naming can be arbitrary
- Updated `__init__.py` for proper Python imports

### 2. Data Augmentations
**Location:** [`dinov2_ssl_8bands/dinov2/data/rs_augmentations.py`](../dinov2_ssl_8bands/dinov2/data/rs_augmentations.py)

- Customized Albumentations-based augmentations for numpy arrays 
- Updated `__init__.py` for proper Python imports

### 3. Vision Transformer Architecture
**Location:** [`dinov2_ssl_8bands/dinov2/models/vision_transformer.py`](../dinov2_ssl_8bands/dinov2/models/vision_transformer.py)

- Modified input channels: `in_chans=8` (from `in_chans=3`) for DinoVisionTransformer(nn.Module)

### 4. Training Pipeline
**Location:** [`dinov2_ssl_8bands/dinov2/train/train_8bands.py`](../dinov2_ssl_8bands/dinov2/train/train_8bands.py)

- Direct execution for pre-training (no SLURM required)
- Configuration example: `configs/ssl_pretraining/SSL_3million.yaml`

### 5. Simple Logging (Weights & Biases)
- Integrated Weights & Biases (wandb) for experiment tracking
- Replace `api_key` with your own account key


## Example Training Run 

### 1. Dataset Preparation
The dataset is stored as `.npy` files inside a single folder.  Each file contains an image-like array with shape `(H, W, 8)` E.g., (256,256,8)
```text
/path/to/dataset/folder/
├── *.npy
├── ...
└── ...
```

### 2. Training 

Run on Multi-GPUs (without SLURM): 

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    /path/to/dinov2_ssl_8bands/dinov2/train/train_8bands.py \
    --output-dir /path/to/output_dir \
    --config-file /path/to/pretraining/config_file.yaml \
    --ssl-data /path/to/dataset/folder \
    --wandb-trial <name_of_the_run> \
    --wandb-project <name_of_the_project>
```
replace the CUDA_VISIBLE_DEVICES and nproc_per_node with specific multi-gpus settings. (e.g., training on 8 A100-80GB GPUs)

Run on Single GPU (without SLURM):
```
python /path/to/dinov2_ssl_8bands/dinov2/train/train_8bands.py \
    --output-dir /path/to/output_dir \
    --config-file /path/to/pretraining/config_file.yaml \
    --ssl-data /path/to/dataset/folder \
    --wandb-trial <name_of_the_run> \
    --wandb-project <name_of_the_project>
```
### 3. Example Traning Logs
Training logs are saved in both image and JSON formats. An example is provided: 

- [Training Metrics Wandb](../configs/ssl_pretraining/training_metrics_wandb.png)  
- [`training_metrics.json`](../configs/ssl_pretraining/training_metrics.json)
