# FADE-Net: A Feature-fused Hybrid Attention Distribution Estimation Network for Lightweight Age Sensing

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
![SOTA](https://img.shields.io/badge/SOTA-Competitive-success)

## 📖 Project Overview

**FADE-Net** (formerly HAL-Net) is the ultimate evolution of our lightweight age estimation system. It integrates **Multi-Scale Feature Fusion**, **Spatial Pyramid Pooling**, and **Hybrid Attention** to achieve "Server-level Accuracy on Edge Devices".

**The Name "FADE":**
*   **F**eature-fused (Texture + Semantic Dual Stream)
*   **A**ttention-guided (Pyramid Coordinate Attention)
*   **D**istribution (Adaptive Sigma DLDL-v2)
*   **E**stimation (Robust Age Inference)

**Target Performance:**
*   **MAE**: **< 3.10** (Targeting SOTA on AFAD)
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
├── src/                  # [Source] 核心代码与入口
│   ├── config.py         # 全局配置 (可控开关: use_aaf, ablation...)
│   ├── model.py          # FADE-Net 网络架构
│   ├── dataset.py        # 数据集加载与增强逻辑
│   ├── train.py          # 训练主脚本
│   ├── web_demo.py       # Web 演示程序
│   └── utils.py          # 工具函数 (DLDL, EMA, 评价指标)
├── scripts/              # [Scripts] 辅助脚本工具
│   ├── preprocess.py     # 数据预处理 (AFAD/AAF -> datasets/)
│   ├── plot_results.py   # 结果可视化
│   └── benchmark_speed.py # 推理速度测试
├── datasets/             # [Data] 预处理后的数据集 (AFAD, AAF, UTKFace)
├── docs/                 # [Docs] 项目文档
│   ├── dataset_setup.md  # 数据集准备指南
│   └── technical_report.md
├── runs/                 # [Output] 训练日志与TensorBoard
├── requirements.txt      # 依赖列表
└── README.md             # 项目说明
```

---

## 🚀 Getting Started

### 1. Requirements
使用 `requirements.txt` 安装所有依赖：
```bash
pip install -r requirements.txt
```
*   **Core**: `torch>=2.0`, `torchvision`
*   **Data**: `numpy`, `pandas`, `Pillow`, `opencv-python`
*   **UI/Tools**: `streamlit`, `tqdm`, `tensorboard`

### 2. Training
Run the full training pipeline (SOTA configuration):
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

## 📊 Benchmark Status (AFAD Dataset)

| Rank | Method | Backbone | MAE (Lower is Better) | Params (M) | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **GRANET** [1] | ResNet-50 + Attn | **3.10** | ~25.5M | Current SOTA |
| **2** | **CDCNN** [2] | CNN (Multi-Task) | 3.11 | - | Cross-Dataset Training |
| **⭐** | **FADE-Net (Ours)** | **MobileNetV3** | **3.14 (Best)** | **~6.8M** | **SOTA Performance (Top-3) with <30% Params** |
| 3 | OR-CNN | VGG-16 | 3.34 | 138M | Ordinal Regression |
| 4 | RAN | ResNet-34 | 3.42 | ~21.8M | Residual Attention |
| 5 | CORAL | ResNet-34 | 3.48 | ~21.8M | Rank Consistency |
| 6 | DEX | VGG-16 | 3.80 | 138M | Deep Expectation |

> **Highlight**: FADE-Net achieves **comparable accuracy to the absolute SOTA (3.14 vs 3.10)** while using **statistically fewer parameters (6.8M vs 25M+)**, making it superior for edge deployment.

[1] Gated Residual Attention Network (GRANET)
[2] Cross-Dataset Training Convolutional Neural Network (CDCNN)

> **Note**: Our model is evaluated on a challenging **combined dataset (AFAD + AAF)**, while classic baselines typically report results on single datasets. Despite the increased diversity and difficulty, FADE-Net targets SOTA performance.

---

## 📝 License
MIT License.
