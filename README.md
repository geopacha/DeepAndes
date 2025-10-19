# 🌎 DeepAndes: A Self-Supervised Vision Foundation Model for Multi-Spectral Remote Sensing Imagery of the Andes  (IEEE JSTARS2025)
> **DeepAndes** is the *first* vision foundation model that applies the **DINOv2** self-supervised learning framework and large-scale pre-training on **multi-spectral satellite imagery** specifically for the **Andes region**.


<a href='https://arxiv.org/abs/2504.20303'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 


## 📢 Latest Updates
🔥 🔥 Last Updated on 2025.10.19 

- **[2025.10.02]** Our paper has been accepted for publication in the IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.


![](assets/figure1.png)


## 📌 Highlights

- **Foundation-model scale**: Trained on ~3 million multi-spectral satellite patches covering ~488,640 km² of the Andes.  
- **Multi-spectral (8-band) input**: Supports 8-band WorldView-2/3 satellite imagery instead of RGB.  
- **Self-supervised learning (SSL)**: Built upon **DINOv2**, adapted for geospatial feature scale and 8-channel inputs.  
- **Downstream versatility**: Evaluated on classification, image-to-image retrieval, and segmentation tasks using the full dataset, as well as on classification and segmentation tasks under few-shot (reduced) dataset settings.

## ⚙️ Architecture & Pre-Training

- **Backbone**: Vision Transformer (ViT-L/16, ~307 M parameters)  
- **Input**: 8-band image patches (<256> × <256>) sampled across diverse Andean terrains  
- **SSL Framework**:  
  - Multi-crop global/local view strategy  
  - Channel adaptation for 8-band input  
  - Large-scale geospatial sampling  
- **Dataset**: ~3M patches across 8 land-cover types (~488 k km² coverage)


## 🚀 Quick Start
<!-- ## 🎯 About This Repository -->
This repo provides 



## 📊 Downstream Tasks


## 📂 Folder Structure
```
deepandes/
├── configs/             # Config files for pre-training & fine-tuning  
├── data/                # Dataset preparation scripts  
├── models/              # Model definitions & checkpoints  
├── downstream/          # Downstream pipelines (classification, retrieval, segmentation)  
├── utils/               # Helper functions (augmentation, logging, etc.)  
├── inference.py         # Simple inference script  
└── train_downstream.py  # Downstream fine-tuning script 
```


## 🦖 Citation
If you find this repository useful, please consider giving a star ⭐ and citation 🦖 Thank you:)

```
@article{guo2025deepandes,
  title={DeepAndes: A Self-Supervised Vision Foundation Model for Multi-Spectral Remote Sensing Imagery of the Andes},
  author={Guo, Junlin and Zimmer-Dauphinee, James R and Nieusma, Jordan M and Lu, Siqi and Liu, Quan and Deng, Ruining and Cui, Can and Yue, Jialin and Lin, Yizhe and Yao, Tianyuan and others},
  journal={arXiv preprint arXiv:2504.20303},
  year={2025}
}
```

## 🧭 Roadmap

- Updating the code for Yolo-Dinov2 object detection head (In-progress)
- 🌎 Extend pre-training to Full Andes Regions (100x more data) (In-progress)
- 🔗 Integrate geospatial metadata and language models 
 


## 🤝 Acknowledgements
Supported by the [GeoPACHA Project](https://geopacha.org/) and collaborators at Vanderbilt University, Brown University, and ORNL.
Special thanks to all contributors from the Andean Archaeology and Remote Sensing communities.

## 📫 Contact & Contribution
For questions or contributions, open an issue or pull request. We are looking forward to your feedback!

Contact: Junlin Guo (junlinguo1@gmail.com), Yuankai Huo (PI)(yuankai.huo@vanderbilt.edu), Steven Wernke (PI)(s.wernke@Vanderbilt.Edu), and Parker VanValkenburgh (parker_vanvalkenburgh@brown.edu)
