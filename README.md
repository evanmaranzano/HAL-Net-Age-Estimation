# FADE-Net: A Feature-fused Hybrid Attention Distribution Estimation Network for Lightweight Age Sensing

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
![Performance](https://img.shields.io/badge/Performance-SOTA_Level-success)

## 📖 Project Overview

**FADE-Net** (formerly HAL-Net) is an **optimized iteration** of our lightweight age estimation system. It integrates **Multi-Scale Feature Fusion**, **Spatial Pyramid Pooling**, and **Hybrid Attention** to achieve "Server-level Accuracy on Edge Devices".

**The Name "FADE":**
*   **F**eature-fused (Texture + Semantic Dual Stream)
*   **A**ttention-guided (Pyramid Coordinate Attention)
*   **D**istribution (Adaptive Sigma DLDL-v2)
*   **E**stimation (Robust Age Inference)

**Target Performance:**
*   **MAE**: **3.01** (Achieves state-of-the-art performance among lightweight models on AFAD in our setting)
*   **Params**: ~6.8M (Lightweight)
*   **Speed**: Real-time on CPU/GPU

---

## ✨ Key Features

1.  **Dual-Stream Architecture (New)**: Defines a "Texture Branch" (Stride-16) and "Semantic Branch" (Stride-32) to capture both fine wrinkles and facial shape.
2.  **Spatial Pyramid Pooling (SPP) (New)**: Enhanced structural design with SPP and stratified splitting further improves representation efficiency.
3.  **Hybrid Attention**: Injecting **Coordinate Attention (CA)** into deep layers to enhance spatial awareness without heavy computation.
4.  **DLDL-v2**: Adaptive Label Distribution Learning with **Ranking Loss (0.3)** and **Weighted L1**.
5.  **Robust Training**: **Mixup** + **Safe Random Erasing** (Synergistic Augmentation) + **Label Sigma Jitter** ensures robust feature learning.
6.  **Fine-Grained Augmentation**: Optimized pipeline with **Affine (Shear/Trans)** and **Gaussian Blur** for geometric and quality robustness.
7.  **Pre-training**: Uses **ImageNet1K V2** weights (Top-1 75.2%) for robust initialization.

---

## 📂 Project Structure

```text
code/
├── src/                  # [Source] Core Logic & Entry Points
│   ├── config.py         # Configuration (Toggles: use_aaf, ablation...)
│   ├── model.py          # FADE-Net Architecture
│   ├── dataset.py        # Dataset Loading & Augmentation
│   ├── train.py          # Main Training Script
│   ├── web_demo.py       # Web Application (Streamlit)
│   └── utils.py          # Utilities (DLDL, EMA, Metrics)
├── scripts/              # [Scripts] Tools & Preprocessing
│   ├── preprocess.py     # Data Preprocessing (AFAD/AAF -> datasets/)
│   ├── plot_results.py   # Visualization
│   └── benchmark_speed.py # Inference Speed Test
├── datasets/             # [Data] Preprocessed Datasets (AFAD, AAF, UTKFace)
├── docs/                 # [Docs] Documentation
│   ├── dataset_setup.md  # Dataset Setup Guide
│   └── technical_report.md
├── runs/                 # [Output] Training Logs & Checkpoints
├── requirements.txt      # Dependencies List
└── README.md             # Project README
```

---

## 🚀 Getting Started

### 1. Requirements
Install dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```
*   **Core**: `torch>=2.0`, `torchvision`
*   **Data**: `numpy`, `pandas`, `Pillow`, `opencv-python`
*   **UI/Tools**: `streamlit`, `tqdm`, `tensorboard`

### 2. Training
Run the full training pipeline (Optimal configuration):
```bash
python src/train.py --epochs 120 --freeze_backbone_epochs 5
```
*   **Checkpoints**: Saved in `checkpoints/`
*   **Logs**: Saved in `runs/FADE-Net_...` (Auto-named based on active modules)

### 3. Evaluation
```bash
python scripts/plot_results.py    # Generate visualization
python src/benchmark_speed.py     # Test FPS
```

---

## 💻 Web Demo
Interactive web interface for real-time age estimation:
```bash
streamlit run src/web_demo.py
```

---

## 📊 Internal Benchmark (AFAD Dataset, Stratified 90-5-5 Split)

| Rank | Method | Backbone | MAE (Lower is Better) | Params (M) | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **FADE-Net (Ours)** | **MobileNetV3** | **3.01 (Our Best)** | **~6.8M** | **SOTA-Level Performance** |
| 2 | **GRANET** [1] | ResNet-50 + Attn | 3.10 | ~25.5M | Previous SOTA |
| 3 | **CDCNN** [2] | CNN (Multi-Task) | 3.11 | - | Cross-Dataset Training |
| 4 | OR-CNN | VGG-16 | 3.34 | 138M | Ordinal Regression |
| 5 | RAN | ResNet-34 | 3.42 | ~21.8M | Residual Attention |
| 6 | CORAL | ResNet-34 | 3.48 | ~21.8M | Rank Consistency |
| 7 | DEX | VGG-16 | 3.80 | 138M | Deep Expectation |

> **Highlight**: FADE-Net achieves **Competitive Accuracy (3.01 vs 3.10)** while using **statistically fewer parameters (6.8M vs 25M+)**, setting a strong baseline for lightweight age estimation.

[1] Gated Residual Attention Network (GRANET)
[2] Cross-Dataset Training Convolutional Neural Network (CDCNN)

> **Note**: Evaluated on AFAD dataset with standard Stratified 90-5-5 Split.

---

## 📝 License
MIT License.
