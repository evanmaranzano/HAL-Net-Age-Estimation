# FADE-Net: Feature-fused Attention Distribution Estimation Network

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
![SOTA](https://img.shields.io/badge/SOTA-Competitive-success)

## 📖 Project Overview

**FADE-Net** (formerly HAL-Net) is the ultimate evolution of our lightweight age estimation system. It integrates **Multi-Scale Feature Fusion**, **Spatial Pyramid Pooling**, and **Hybrid Attention** to achieve "Server-level Accuracy on Edge Devices".

**The Name "FADE":**
*   **F**eature-fused (Texture + Semantic Dual Stream)
*   **A**ttention-guided (Pyramid Coordinate Attention)
*   **D**istribution (Adaptive Sigma DLDL-v2)
*   **E**nhanced Structure (SPP + Stratified Split)

**Target Performance:**
*   **MAE**: **< 3.10** (Targeting SOTA on AFAD)
*   **Params**: ~6.8M (Lightweight)
*   **Speed**: Real-time on CPU/GPU

---

## ✨ Key Features

1.  **Dual-Stream Architecture (New)**: Defines a "Texture Branch" (Stride-16) and "Semantic Branch" (Stride-32) to capture both fine wrinkles and facial shape.
2.  **Spatial Pyramid Pooling (SPP) (New)**: Replaces global pooling with 1x1, 2x2, 4x4 adaptive pooling to preserve spatial layout information.
3.  **Hybrid Attention**: Injecting **Coordinate Attention (CA)** into deep layers to enhance spatial awareness without heavy computation.
4.  **DLDL-v2**: Adaptive Label Distribution Learning with **Ranking Loss** and **LDS** (Label Distribution Smoothing) to handle label ambiguity.
5.  **Pre-training**: Uses **ImageNet1K V2** weights (Top-1 75.2%) for robust initialization.

---

## 📂 Project Structure

```text
├── config.py             # [Core] Global configuration (Hyperparams, Ablation Flags)
├── model.py              # [Core] FADE-Net architecture (Backbone + SPP + Fusion)
├── dataset.py            # [Data] Dataset class & Stratified Splitting
├── train.py              # [Main] Training loop with Freeze Strategy
├── utils.py              # [Utils] DLDL-v2 Loss, EMA, Metrics
├── technical_report.md   # [Docs] Detailed Technical Report
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### 1. Requirements
```bash
pip install torch torchvision numpy pandas tqdm tensorboard matplotlib scipy
```

### 2. Training
Run the full training pipeline (SOTA configuration):
```bash
python train.py --epochs 120 --freeze_backbone_epochs 5
```
*   **Checkpoints**: Saved in `checkpoints/`
*   **Logs**: Saved in `runs/FADE-Net_...` (Auto-named based on active modules)

### 3. Evaluation
```bash
python plot_results.py    # Generate visualization
python benchmark_speed.py # Test FPS
```

---

## 💻 Web Demo
Interactive web interface for real-time age estimation:
```bash
streamlit run web_demo.py
```

---

## 📊 Benchmark Status
| Model | Backbone | Params | MAE (Lower is Better) | Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **FADE-Net (Ours)** | **MobileNetV3** | **~6.8M** | **Targeting < 3.10** | **AFAD + AAF (Combined)** |
| ResNet-18 | ResNet-18 | 11.7M | ~3.11 | AFAD (Single) |
| OR-CNN | VGG-16 | 138M | 3.34 | AFAD (Single) |

> **Note**: Our model is evaluated on a challenging **combined dataset (AFAD + AAF)**, while classic baselines typically report results on single datasets. Despite the increased diversity and difficulty, FADE-Net targets SOTA performance.

---

## 📝 License
MIT License.
