# Multi-band Imagery Support for DINOv2
The DINOv2 framework was originally designed for **RGB** images (3 bands). However, in fields like remote sensing and earth observation, we often need **multispectral** or **hyperspectral** imagery (>3 bands).

## Purpose

- Extends Meta's DINOv2 (and potentially DINOv3) to support multispectral imagery pre-training (e.g., 8-band [WorldView-2](https://www.satimagingcorp.com/satellite-sensors/worldview-2/) and [WorldView-3](https://www.satimagingcorp.com/satellite-sensors/worldview-3/) satellite imagery) 
- Maintain modularity while minimizing changes.

## Installation 
This guide creates a Python 3.10 virtual environment with the necessary dependencies for training DINOv2 with 8-band support and custom augmentations.

1. Create a Conda Environment
```
conda create -n dinov2_env python=3.10
conda activate dinov2_env
```

2. Install PyTorch 
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```
🔁 Replace cu128 with cu118, cu117, etc., depending on your system. The command/instructions can be found from [Pytorch](https://pytorch.org/get-started/locally/).

3. Install Core Libraries: evaluation metrics, configuration parser, FAIR utilities, optimized attention layers, and experiment tracking.

```bash
pip install torchmetrics  
pip install omegaconf    
pip install fvcore iopath 
pip install xformers     
pip install wandb        
```


4. Install Albumentations (with Pydantic v2)
```
pip install albumentations==1.4
pip install -U pydantic
```
⚠️ Attention: The higher version (>2.0+) of albumentations maybe conflict with pydantic v2. So installed the albumentations==1.4 here.

5. (Optional) Save the Environment

```
# Save pip-based environment
pip freeze > requirements.txt

# Or export full Conda environment
conda env export --no-builds > dinov2_env.yaml

```
[requirement.txt](../configs/ssl_pretraining/conda_envs/requirements.txt) and [dinov2_env.yaml](../configs/ssl_pretraining/conda_envs/dinov2_env.yaml) are saved.

## Example Training Run 

### 1. Dataset Preparation
The dataset for pre-training is stored as `.npy` files inside a single folder.  Each file contains an image-like array with shape `(H, W, C)` (E.g., H=W=256, C=8 in our case). 
```text
/path/to/dataset/folder/
├── *.npy
├── ...
└── ...
```
### 2. Training Config

An example config file for training on 3 million image patches (prepared as Step 1) is provided in `../configs/ssl_pretraining`: [SSL_3million.yaml](../configs/ssl_pretraining/SSL_3million.yaml). 

### 3. Training 

Adjust Path and Key: `dinov2_ssl_8bands/dinov2/train/train_8bands.py`
```python
import sys
import os 

# Replace '/path/to/dinov2_ssl_8bands' with your actual path
# which is necessary for Python to locate the `dinov2` package during training.
sys.path.append('/path/to/dinov2_ssl_8bands')


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    wandb.login(key="api_key_here") # Replace with your wandb api_key
```

Run on Multi-GPUs (without SLURM): 

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    /path/to/dinov2_ssl_8bands/dinov2/train/train_8bands.py \
    --output-dir /path/to/output_dir \
    --config-file /path/to/config_file.yaml \
    --ssl-data /path/to/dataset/folder \
    --wandb-trial <name_of_the_run> \
    --wandb-project <name_of_the_project>
```
replace the CUDA_VISIBLE_DEVICES and nproc_per_node with specific multi-gpus settings. (e.g., training on 8 A100-80GB GPUs)

Run on Single GPU (without SLURM):
```
python /path/to/dinov2_ssl_8bands/dinov2/train/train_8bands.py \
    --output-dir /path/to/output_dir \
    --config-file /path/to/config_file.yaml \
    --ssl-data /path/to/dataset/folder \
    --wandb-trial <name_of_the_run> \
    --wandb-project <name_of_the_project>
```
### 4. Example Traning Logs
An example training logs is provided here:: 

- [training_metrics (wandb snapshot)](../configs/ssl_pretraining/training_metrics_wandb.png)  
- [training_metrics (json format)](../configs/ssl_pretraining/training_metrics.json)

After pre-training, the model checkpoints can be found in `path/to/output_dir/eval/training_[number]/teacher_checkpoint.pth`

## Summary of Key Modifications

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
- Configuration example: [SSL_3million.yaml](../configs/ssl_pretraining/SSL_3million.yaml)

### 5. Simple Logging (Weights & Biases)
- Integrated Weights & Biases (wandb) for experiment tracking
- Replace `api_key` with your own account key

## How We Verify the Same Implementation of Data Loading
Both `train.py` (RGB) and `train_8bands.py` (8-band) use **identical data loading strategies and randomness methods**. The only differences are the dataset class and augmentation class.

| Aspect | `train.py` (RGB) | `train_8bands.py` (8-band) |
|--------|-----------------|---------------------------|
| **Dataset class** | `ImageNet` (via `make_dataset`) | `NLBDataset` (directly instantiated) |
| **Augmentation** | `DataAugmentationDINO` (RGB torchvision transforms) | `DataAugmentationDINO_MS` (albumentations for multi-spectral) |
| **Sampler type** | `SHARDED_INFINITE` | `SHARDED_INFINITE` — identical |
| **shuffle** | `True` | `True` — identical |
| **seed** | `seed=start_iter` | `seed=start_iter` — identical |
| **sampler_advance** | `0` | `0` — identical |
| **collate_fn** | `collate_data_and_cast` | `collate_data_and_cast` — identical |
| **drop_last** | `True` | `True` — identical |
| **target_transform** | `lambda _: ()` (discards labels) | not passed; `NLBDataset.get_target()` always returns `0` |


## Citing Our Work

If you find this repository useful, please consider giving a star ⭐ and citation 🦖 Thank you:)

```
@article{guo2025deepandes,
  title={DeepAndes: A Self-Supervised Vision Foundation Model for Multi-Spectral Remote Sensing Imagery of the Andes},
  author={Guo, Junlin and Zimmer-Dauphinee, James R and Nieusma, Jordan M and Lu, Siqi and Liu, Quan and Deng, Ruining and Cui, Can and Yue, Jialin and Lin, Yizhe and Yao, Tianyuan and others},
  journal={arXiv preprint arXiv:2504.20303},
  year={2025}
}
```

