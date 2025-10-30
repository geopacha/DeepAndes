# 🧩 Installation

Follow the steps below to set up the environment for this project.

## 1. Create a Conda Environment

```bash
conda create -n faiss_py10 python=3.10
conda activate faiss_py10
```
⚠️ Note: Python 3.10 (>3.9) is required for torch.hub to properly load model.

## 2. Install Required Packages 
The `faiss-gpu` (v1.7.2) library requires `numpy` version < 2.0 and is validated on PyTorch CUDA 12.1.

```bash
pip install tqdm
pip install matplotlib
pip install faiss-gpu==1.7.2
pip install -U albumentations==1.4.22
pip install numpy==1.26.3

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install timm
```

## ✅ Verify Installation
```bash
python -c "import torch, timm, faiss, cv2; print(torch.__version__, timm.__version__)"
```