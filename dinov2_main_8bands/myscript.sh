#!/bin/bash

# Set the root directory of your project as PYTHONPATH
export PYTHONPATH=/workspace/geopacha/dinov2-main:$PYTHONPATH

# Set the CUDA devices to be used
export CUDA_VISIBLE_DEVICES=all

# Run the Python script with configuration file and output directory
echo "Running the DINOv2 training script..."
python dinov2/train/train_rs.py \
—config-file dinov2/configs/train/vitl16_short_copy.yaml \
—output-dir dinov2/output_train

echo "Script execution completed."
