# Multi-band Imagery Support for DINOv2

## Overview

The DINOv2 framework was originally designed for RGB images (3 bands). However, in fields like remote sensing, archaeology, and earth observation, we often need to work with multispectral or hyperspectral imagery (>3 bands) to capture more information.

## Purpose

This modification extends Meta's DINOv2 (and potentially DINOv3) to support multispectral imagery pre-training (e.g., 8-band satellite imagery) while minimizing changes and maintaining modularity.

## Key Modifications

### 1. Dataset Module
**Location:** `dinov2_ssl_8bands/dinov2/data/datasets/`

- Added `nlb_dataset.py` module for 8-band data stored as individual `.npy` files
- File naming is unconstrained
- Extends the existing `extended.py` module
- Updated `__init__.py` for proper Python imports

### 2. Data Augmentations
**Location:** `dinov2_ssl_8bands/dinov2/data/rs_augmentations.py`

- Customized Albumentations-based augmentations for numpy arrays
- Easily adaptable for different input types
- Updated `__init__.py` for proper Python imports

### 3. Vision Transformer Architecture
**Location:** `dinov2_ssl_8bands/dinov2/models/vision_transformer.py`

- Modified input channels: `in_chans=8` (from `in_chans=3`)

### 4. Training Pipeline
**Location:** `dinov2_ssl_8bands/dinov2/train/train_8bands.py`

- Direct execution for pre-training (no SLURM required)
- Configuration example: `configs/ssl_pretraining/SSL_3million.yaml`

### 5. Logging (Optional)
- Integrated Weights & Biases (wandb) for experiment tracking
- Replace `api_key` with your own account key