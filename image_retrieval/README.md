# Image to Image Retrieval (IIR)

This task evaluates the model’s ability to retrieve semantically similar archaeological loci image **without additional fine-tuning**. Given a query image, the system ranks database images by cosine similarity (using the class token) and returns the top-k matches.  

This approach enables rapid dataset expansion — starting from a small set of labeled loci image, it can automatically discover and group related samples across unexplored regions.


## Quick Start

1. **Install Environment**: Follow the setup instructions in [Installation](#installation-conda)
2. **Prepare Dataset**: Organize your `.npy` image files as described in [Dataset Preparation](#dataset-preparation)
3. **Run Notebook**: Open [deepandes_feature_extract.ipynb](deepandes_feature_extract.ipynb) and configure your paths. Similar settings apply for other baseline notebooks.
4. **Evaluate IIR**: An example ([deepandes_IIR_evaluate.ipynb](deepandes_IIR_evaluate.ipynb)) demonstrates mean average precision (mAP) calculation for Top-K positive image retrieval.



## Installation (Conda)
We use the [FAISS](https://github.com/facebookresearch/faiss) library for fast, simple image-to-image retrieval. To set up the Conda environment, follow the step-by-step install instructions in [faiss_install.md](faiss_install.md). 

As a reference, both [requirements.txt](requirements.txt) and [environment.yaml](faiss_py10.yaml) are provided for environment setup, check package versions if needed.



## Dataset Preparation 

All images (i.e., database) are stored as `.npy` files inside a single folder. Each image (`numpy array`) with 8 spectral bands/channels, having a shape of `(256, 256, 8)` and a data type of `np.uint8`. 

**Basic Structure:**
```text
/path/to/dataset/folder/
├── *.npy
├── ...
└── ...
```

**For IIR Evaluation (Binary Classification):**
In our paper, we evaluate the mean average precision (mAP) of the IIR task. Particularly, the dataset used is the binary archaeological loci classification dataset (`CLS0` or `CLS1`)used for our previous binary loci classification task, organized as:

```text
/path/to/dataset/folder/
├── CLS0-*.npy
├── CLS0-*.npy
├── CLS1-*.npy
├── CLS0-*.npy
├── CLS1-*.npy
├── ...
└── ...
```

## Implementations 

An example for zero-shot image-to-image retrieval using the DeepAndes pre-trained backbone is provided: [deepandes_feature_extract.ipynb](deepandes_feature_extract.ipynb). Some parameters are defined.

**Configuration Parameters:**

```python
pretrained_weight = '/path/to/teacher_checkpoint.pth'  # Path to pre-trained checkpoint
device = torch.device("cuda:0")                        # Specify GPU index 
path_to_all_images = '/path/to/dataset/folder/*.npy'   # Path to all database images
number_to_retrieve = 10                                # Top-k retrieval by cosine similarity

query_image_path = '/path/to/CLS1-7760-223.npy'        # A example query loci image for retrieval
```
In our work, we display the image using channels/bands 4, 2, and 1 as RGB for visualization purposes only. The example below shows a query image (Class 1, **active corrals** — dark areas indicate animal use) and the top-10 retrieved images based on cosine similarity.

![](../assets/retrieval.png)
**Typical Workflow:**
1. Load the pre-trained (DeepAndes/other) model
2. Extract features from all database images
3. Build a FAISS index for efficient similarity search
4. Query with a test image and retrieve top-k results


### Troubleshooting

If the error `ModuleNotFoundError: No module named 'dinov2.hub.dinotxt'` occurs while loading module, simply comment out the following line in the `~/.cache/torch/hub/facebookresearch_dinov2_main/hubconf.py` file:

```python
# from dinov2.hub.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l
```
An example [hubconf.py](../configs/hubconf.py) is provided. 
This is the config mis-match since we use the simple torch hub loading and adjust the pre-trained wieght. 


## Other Baseline Models Comparison

We also prototyped other self-supervised learning baselines for image retrieval:

| Model | Description | Notebook |
|-------|-------------|----------|
| **DeepAndes** | Proposed multi-spectral self-supervised foundation model | [deepandes_feature_extract.ipynb](deepandes_feature_extract.ipynb) |
| **MAE** | Masked Autoencoder | [mae_feature_extract.ipynb](mae_feature_extract.ipynb) |
| **MoCo-V2** | Momentum Contrast v2 | [mocov2_feature_extract.ipynb](mocov2_feature_extract.ipynb) |
| **SatMAE** | Satellite Masked Autoencoder baseline | [satmae_feature_extract.ipynb](satmae_feature_extract.ipynb) |
| **Scratch** | Randomly initialized ViT-L (no pre-training) | [scratch_feature_extract.ipynb](scratch_feature_extract.ipynb) |


## Evaluate IIR with mean Average Precision (mAP) 
An example ([deepandes_IIR_evaluate.ipynb](deepandes_IIR_evaluate.ipynb)) demonstrates mean average precision (mAP) calculation for Top-K positive image retrieval. The same script structure and helper functions can be applied to other baseline evaluations.
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
<br>

## Contact 

For questions or contributions, open an issue or pull request. We are looking forward to your feedback!

Contact: Junlin Guo (junlinguo1@gmail.com), Yuankai Huo (PI)(yuankai.huo@vanderbilt.edu), Steven Wernke (PI)(s.wernke@Vanderbilt.Edu), and Parker VanValkenburgh (parker_vanvalkenburgh@brown.edu)
