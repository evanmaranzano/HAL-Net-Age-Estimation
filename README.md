# HAL-Net: Hybrid Attention Lightweight Age Estimation

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
![SOTA](https://img.shields.io/badge/SOTA-Competitive-success)

## 📖 Project Overview (项目概述)

This project implements **HAL-Net** (Hybrid Attention Lightweight Network), a high-performance age estimation system optimized for edge devices. It achieves state-of-the-art (SOTA) level accuracy on the AFAD dataset using **MobileNetV3-Large** combined with **Coordinate Attention**, **Deep Label Distribution Learning (DLDL-v2)** and strict **Stratified Sampling**.

**Key Performance Indicators:**
*   **MAE**: **3.1480** (Test SOTA Competitive)
*   **Inference Speed**: **122 FPS** (RTX 3060), **59 FPS** (Ryzen 9 CPU)
*   **Parameters**: ~5.4M
*   **FLOPs**: ~219M

---

## ✨ Key Features (核心特性)

1.  **Lightweight Backbone**: Built on `MobileNetV3-Large` for optimal speed/accuracy trade-off.
2.  **Hybrid Attention (New)**: Incorporates **Coordinate Attention (CA)** in deep layers (Stage 4-5) to capture spatial aging features (wrinkles, face shape) while keeping shallow layers efficient.
3.  **DLDL-v2 (Deep Label Distribution Learning)**: Enhanced DLDL with **Adaptive Sigma**, **Ranking/CDF Loss**, and **LDS** (Label Distribution Smoothing) to handle label ambiguity and imbalance.
4.  **Stratified Sampling**: Implements a rigorous `90/5/5` split based on age distribution to ensure validating on representative data.
5.  **Freeze Training Strategy**: Protects pre-trained backbone features during the initial phase of training (Warm-up + Freeze).
6.  **Advanced Reg**: Incorporates `MixUp` (alpha=0.2), `Dynamic Dropout`, and `EMA` (Exponential Moving Average) for robust generalization.

---

## 📂 Project Structure (目录结构)

```text
├── config.py             # [Core] Global configuration (Hyperparams, Paths)
├── model.py              # [Core] Model architecture definition (MBV3 + Custom Head)
├── dataset.py            # [Data] Dataset class, Loading, and Stratified Splitting
├── train.py              # [Main] Training loop, Validation, Checkpointing
├── utils.py              # [Utils] Loss functions (KL+L1+Rank), DLDL logic, EMA
├── benchmark_speed.py    # [Tools] Interface speed benchmarking (FPS/Latency)
├── plot_results.py       # [Tools] Generate training visualization plots
└── README.md             # Project documentation
```

---

## 🚀 Getting Started (快速开始)

### 1. Requirements
```bash
pip install torch torchvision numpy pandas tqdm tensorboard matplotlib scipy
```

### 2. Configure Paths
Verify your dataset paths in `config.py`:
```python
# config.py
self.afad_dir = "./data_aligned/AFAD"
self.aaf_dir = "./data_aligned/AAF"
```

### 2.5. Data Preprocessing (数据预处理)
Use `preprocess.py` to align faces and perform stratified splitting:
```bash
python preprocess.py
```
*   **Align Faces**: Detects and aligns faces from source datasets (AFAD/AAF).
*   **Stratified Split**: Generates `dataset_split_stratified.json` with 90/5/5 ratio.

### 3. Training (训练)
Start the training process with SOTA presets:
```bash
python train.py
```
*   **Outputs**:
    *   `best_model.pth`: Model with lowest Val MAE.
    *   `training_log.csv`: Detailed epoch-wise metrics.
    *   `runs/`: TensorBoard logs.

### 4. Evaluation & Visualization (评估与可视化)
Generate performance plots (Loss, MAE, LR Schedule):
```bash
python plot_results.py
```
Run hardware benchmark:
```bash
python benchmark_speed.py
```

---

## 💻 Web Demo (可视化演示)

Run the interactive web interface for age estimation:
```bash
streamlit run web_demo.py
```
**Features:**
*   **Single Image Analysis**: Upload or take a snapshot to estimate age with uncertainty plots.
*   **Batch Processing**: Process multiple images at once and export results to CSV.
*   **Real-time Video**: Live age estimation from webcam feed.

---

## 📊 Benchmark Results (AFAD Dataset)

| Model | Backbone | Params | MAE (Lower is Better) |
| :--- | :--- | :--- | :--- |
| **Ours** | **MobileNetV3** | **5.4M** | **3.1480** |
| ResNet-18 | ResNet-18 | 11.7M | ~3.11 |
| GhostNet | GhostNet | 5.2M | N/A (Theoretical) |
| OR-CNN | VGG-16 | 138M | 3.34 |

---

## 📝 License
This project is open-source and available under the MIT License.
