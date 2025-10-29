# Classification Evaluation

We evaluate the pre-trained backbones on a binary classification task using a two-layer linear classifier, providing a simple yet effective benchmark for assessing the feature quality of the pre-trained backbone. 

**Content Summary**

- How the classification dataset is structured and formatted
- How to launch the training, enable experiment tracking
- SSL Baseline Models Comparison: **DeepAndes**, **MAE**, **MoCo-v2**, **SATMAE**, and **Scratch**

This linear probe evaluation can be modified very flexibly. The key is to load the pre-trained backbone and append a classifier. An example script is provided ``linear_prob_simple_args.py``

## Installation (Conda)
Linear classification can use the same conda environment as pre-training; refer to the dinov2_8bands [**installation**](../dinov2_ssl_8bands/README.md#installation). When using the `linear_prob_simple_args.py` module, install the required libraries:

```bash
conda activate dinov2_env

pip install matplotlib pandas timm
```
These libraries are needed for data analysis and for easier loading of other baseline backbones.


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

## Training CLI

After pre-training (see [SSL README](../dinov2_ssl_8bands/README.md)), checkpoints are saved at: `/path/to/output_dir/eval/`. We provided our pre-trained ViT-L/14 backbone on [Google Drive](https://drive.google.com/drive/folders/1-9XMSWyto_-3Rh7U4ObdjhgETvZkZ9PD?usp=sharing).


Adjust Path and Key
```python
import sys

# Replace '/path/to/dinov2_ssl_8bands' with your actual path
sys.path.append('/path/to/dinov2_ssl_8bands')

if use_wandb:
    wandb.login(key="api_key_here") # Replace with your wandb api_key
```

To fine-tune a model (e.g., `deepandes`) using binary classification dataset, run:

```
python ./classification_eval/linear_prob_simple_args.py \
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
Replace each placeholder (like `<wandb_project_name>`) as appropriate.



### Other Baseline Models Comparison

The `--model_name` flag supports the following backbone options:

- `deepandes` — our ViT-L model from DINOv2
- `mae` — Masked Autoencoder
- `mocov2` — Momentum Contrast v2
- `satmae` — Other Satellite MAE baseline
- `scratch` — randomly initialized ViT-L (no pre-training)


<br>
To fine-tune MAE backbone:

```
python ./classification_eval/linear_prob_simple_args.py \
    --use_wandb \
    --wandb_project <wandb_project_name> \
    --wandb_trial <wandb_run_name> \
    --train_dataset_str /path/to/train_dataset_dir \
    --val_dataset_str /path/to/val_dataset_dir \
    --output_dir /path/to/output_dir \
    --epochs 10 \
    --model_name mae
```


<br>
To fine-tune MoCo-V2 backbone: 

```
python ./classification_eval/linear_prob_simple_args.py \
    --use_wandb \
    --wandb_project <wandb_project_name> \
    --wandb_trial <wandb_run_name> \
    --train_dataset_str /path/to/train_dataset_dir \
    --val_dataset_str /path/to/val_dataset_dir \
    --output_dir /path/to/output_dir \
    --epochs 10 \
    --model_name deepandes \
    --pretrained_weights /path/to/moco_v2_200ep_pretrain.pth.tar
```

the moco pre-trained weight can be downloaded from offical github [download here](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar).


<br>
To fine-tune SatMAE backbone:

```
python ./classification_eval/linear_prob_simple_args.py \
    --use_wandb \
    --wandb_project <wandb_project_name> \
    --wandb_trial <wandb_run_name> \
    --train_dataset_str /path/to/train_dataset_dir \
    --val_dataset_str /path/to/val_dataset_dir \
    --output_dir /path/to/output_dir \
    --epochs 10 \
    --model_name satmae
```


<br>
To fine-tune ViT-L/14 backbone with no pretrained weights (Scratch):

```
python ./classification_eval/linear_prob_simple_args.py \
    --use_wandb \
    --wandb_project <wandb_project_name> \
    --wandb_trial <wandb_run_name> \
    --train_dataset_str /path/to/train_dataset_dir \
    --val_dataset_str /path/to/val_dataset_dir \
    --output_dir /path/to/output_dir \
    --epochs 10 \
    --model_name scratch
```
<br>

**Notes**: Public SSL backbones for comparison are adapted to 8 bands (by adjusting patch embedding) using the `timm` API, which also supports the DINO series and other SOTA PyTorch-based ViT models. An example of this adjustment is [moco_loader.py](../dinov2_ssl_8bands/dinov2/eval/other_baselines/moco_loader.py)


<br>

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
