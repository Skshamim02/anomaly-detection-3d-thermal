# Thermal Anomaly Detection in 3D Printing

This project implements deep learning models (CNNs, Vision Transformers, and Autoencoders) for detecting thermal anomalies in 3D printing processes using thermal image data.

## 🔍 Overview

Thermal anomalies during 3D printing can lead to structural defects in printed objects. This project explores various supervised, unsupervised, and semi-supervised approaches to identify such anomalies in real-time using computer vision and deep learning techniques.

## 🛠️ Methods

- **Backbones**: WideResNet, ResNet, Vision Transformers (ViT)
- **Approaches**:
  - Supervised Learning
  - Unsupervised Anomaly Detection
  - Semi-Supervised Learning
- **Architectures**:
  - Convolutional Neural Networks (CNNs)
  - Autoencoders
  - Feature Embedding Models

## 📁 Dataset

- **WFDD (Woven Fabric Defect Detection)** for anomaly patterns.
- **DTD (Describable Textures Dataset)** as background augmentation.
- Custom split for categories like `grey_cloth`, `grid_cloth`, etc.

Folder structure:
