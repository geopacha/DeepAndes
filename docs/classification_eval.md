# Classification Evaluation

We evaluate the pretrained model backbones on a binary classification task using a lightweight linear classifier composed of two fully connected (FC) layers. This serves as a simple yet effective downstream benchmark to assess the feature quality of the pretrained backbone.



Specifically, it covers:

- How the classification dataset is structured and formatted
- How to launch the training script, enable experiment tracking
- Available backbone options adapted to 8-band satellite imagery, including:
   **DeepAndes**, **MAE**, **MoCo-v2**, **SATMAE**, and **Scratch**

------

## Dataset Format

Each image is saved as a `.npy` file with 8 spectral bands/channels, having a shape of `(256, 256, 8)` and a data type of `np.uint8`. The  structure follows the standard used by torchvision:

```
/path/to/train_dataset_dir/
    ├── 0/  # Negative samples
    │   └── *.npy
    └── 1/  # Positive samples
        └── *.npy

/path/to/val_dataset_dir/
    ├── 0/  # Negative samples
    │   └── *.npy
    └── 1/  # Positive samples
        └── *.npy
```

------

## WandB Integration

To enable experiment tracking with Weights & Biases, include the `--use_wandb` flag in your CLI command and initialize your API key:

```
wandb.login(key="your_wandb_api_key_paste_here")
```

------

## Training CLI

To fine-tune a model (e.g., `deepandes`) using classification dataset, run:

```
python /path/to/classification_eval/linear_prob_simple_args.py \
    --use_wandb \
    --wandb_project <wandb_project_name> \
    --wandb_trial <wandb_run_name> \
    --train_dataset_str /path/to/train_dataset_dir \
    --val_dataset_str /path/to/val_dataset_dir \
    --output_dir /path/to/output_dir \
    --epochs 10 \
    --model_name deepandes \
    --pretrained_weights /path/to/teacher_checkpoint.pth
```

>  After pretraining (e.g., [SSL pretraining README](./adjust_pretrain_for_8bands.md)), checkpoints are typically saved under:
>  /path/to/output_dir/eval/training_[number]/teacher_checkpoint.pth`

Replace each placeholder (like `<your_project_name>`) as appropriate.

------

### Other Model Options

The `--model_name` flag supports the following backbone options:

- `deepandes` — our ViT-L model from DINOv2
- `mae` — Masked Autoencoder
- `mocov2` — Momentum Contrast v2
- `satmae` — Other Satellite MAE baseline
- `scratch` — randomly initialized ViT-L (no pretraining)