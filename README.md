# 🔥 Anomaly Detection in 3D Thermal Printing

This repository contains the implementation of a research project focused on thermal anomaly detection in 3D printing using deep learning. It is inspired by the GLASS framework and builds upon it with custom training strategies, a WideResNet-50 backbone, and real 3D thermal print data.

We explore and compare supervised, unsupervised, and semi-supervised approaches using both standard and custom anomaly synthesis methods. The project aims to contribute toward automated quality monitoring in additive manufacturing.

---

## 📂 Project Highlights

- 🔍 Thermal image anomaly detection for 3D printing
- 🧠 Deep learning with WideResNet-50 backbone
- 🧪 Comparative study of supervised, unsupervised, and semi-supervised methods
- 🎛️ Support for contrastive learning and meta-learning training strategies
- ⚙️ Dataset support for WFDD (fabric defects) and DTD (texture augmentation)

---

## 🚀 Getting Started

Clone the repository and install dependencies:

```bash

git clone https://github.com/Skshamim02/anomaly-detection-3d-thermal.git
cd anomaly-detection-3d-thermal
pip install -r requirements.txt
Note: If requirements.txt is not available, manually install dependencies listed in environment.yml or use a compatible PyTorch 1.12+ setup with NVIDIA GPU.


```
📁 Dataset
This project uses modified subsets of the following datasets:

WFDD (Woven Fabric Defect Detection): Four categories including grey_cloth, grid_cloth, yellow_cloth, pink_flower

DTD (Describable Textures Dataset): Used for background/texture augmentation

You can replace these with your own thermal anomaly dataset if available

Download the WFDD dataset from Google Drive

⚙️ Training & Inference
Run training with your desired configuration (example below):

```bash
python main.py \
    --gpu 0 \
    --test ckpt \
    --net wideresnet50 \
    --datapath ./data \
    --augpath ./data/dtd/images \
    --dataset wfdd \
    --class yellow_cloth \
    --output results \
    --alpha 0.5 \
    --fg 1 \
    --epochs 200
```
Change --test to test or train to switch modes.


🧠 Model & Strategy
We currently support:

✅ WideResNet-50 backbone (default)

🔁 Noisy patch embedding strategy

🌌 Feature mining and gradient ascent augmentation

🔍 Contrastive loss with configurable radius

📦 Custom patch sampling: normal, abnormal, and hard negatives

You can modify these strategies in model/ and trainer/ directories.

📊 Results & Evaluation
Results are saved under the results/ directory. Use visualization scripts to generate heatmaps and bounding box overlays. Metrics include:

AUROC (Area Under ROC Curve)

Pixel-level and image-level F1 scores

Heatmap visualizations of anomaly localization

📌 Future Work
 Integration with real-time thermal camera feed

 Comparison with Transformer-based architectures

 Extension to semantic anomaly segmentation

 Autoencoder baseline integration

🙏 Acknowledgements
This project builds on the GLASS repository and was inspired by concepts from:
SimpleNet
BGAD
MVTec AD
