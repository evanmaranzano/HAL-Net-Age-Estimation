# Training Configuration & Results Log (Seed 2027c)

**Edition**: Architecture Evolution (v2.1)
**Date**: 2026-01-10
**Status**: Ready for Launch (v3)

## 1. Parameters Snapshot (2027c Optimization)

### Core Training
- **Epochs**: 120
- **Batch Size**: 128
- **Learning Rate**: 0.0003 (3e-4) -> Cosine Annealing
- **Weight Decay**: 1e-4
- **Freeze Backbone**: First 10 Epochs
- **Cooldown Phase**: Epoch >= 105
  - **Re-Loader Strategy**: Re-builds DataLoader to flush persistent workers (Worker-Safe).
  - **Transforms**: Switches to pure Validation Transform (Clean Images).
  - **Augmentations**: Strictly disables Mixup, RE, and Sigma Jitter.

### Loss Function (CombinedLoss)
| Component | Weight | Note |
| :--- | :--- | :--- |
| **KL Divergence** | 1.0 | Base distribution loss |
| **L1 Loss** | 0.1 | Auxiliary absolute error |
| **Rank Loss** | **0.5** | Balanced ordinal constraint |
| **Mean-Variance** | 0.1 | Normalized by Range² |

### Model Architecture
- **Backbone**: MobileNetV3
- **Features**: 
  - [x] Hybrid Attention (CoordAtt)
  - [x] DLDL v2 (Adaptive Sigma)
  - [x] Multi-Scale Feature Fusion (MSFF)
  - [x] **Bottleneck SPP v2**: 5/9/13 scales + 1x1 Projection (512-ch).

### Data & Augmentation
- **Split**: 80-10-10 (Standard Stability Protocol)
- **Mixup**: 0.5 (Disabled in Cooldown)
- **Dropout**: 0.1

## 2. Result Recording

| Seed | Split | Test MAE | Best Val MAE | Epoch (Best) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2027c** | 80-10-10 | | | | Architecture Evolution |

---
*Optimized & Hardened by Antigravity*
