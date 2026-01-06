import os
import torch

# Define Project Root (src is one level deep)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    # --- 1. 🔬 Ablation Switch (消融实验核心开关) ---
    use_hybrid_attention = True  # HA: Coordinate Attention
    use_dldl_v2 = True           # DLDL: Adaptive Sigma + Rank Loss
    use_multi_scale = True       # MSFF: Texture-Semantics Dual-Stream
    use_spp = True               # SPP: Spatial Pyramid Pooling
    
    # --- 1.1 📊 Split Protocol (New) ---
    # Options: '90-5-5' (Our Best) or '72-8-20' (Standard 80-20 implementation)
    split_protocol = '90-5-5'

    # --- 1.2 🌱 Academic Seeds (with Meanings) ---
    ACADEMIC_SEEDS = {
        42:   "The Answer to Life, the Universe, and Everything",
        3407: '"Torch.manual_seed(3407) is all you need" (arXiv:2109.08203)',
        2026: "Current Year (Modernity Check)",
        1337: "Leet (Elite)",
        1106: "Special Dedication <3 (Randomly Sampled w.r.t our hearts)"
    }

    # --- 2. 🚀 动态项目命名逻辑 (Robust & Dynamic) ---
    @property
    def project_name(self):
        base = "FADE-Net"
        tags = []
        if self.use_hybrid_attention: tags.append("HA")
        if self.use_dldl_v2:          tags.append("DLDL")
        if self.use_multi_scale:      tags.append("MSFF")
        if self.use_spp:              tags.append("SPP")
        
        suffix = "_".join(tags) if tags else "Baseline"
        return f"{base}_{suffix}"

    # --- 3. 🎯 核心超参数 (Based on Final Tuning) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 年龄区间
    min_age, max_age = 0, 80
    num_classes = 81 
    
    # DLDL-v2 动态微调参数
    use_adaptive_sigma = True
    sigma_min = 1.0              #  Revert to 1.0 for stability (Mixup Removed)
    sigma_max = 3.5
    lambda_l1 = 0.1              # 📉 Plan C: Downgrade to auxiliary
    lambda_rank = 1.0            # 👑 Plan C: Rank is King (BCE Version)
    
    # Label-Level Perturbation (Sigma Jitter)
    use_sigma_jitter = True
    sigma_jitter = 0.2
    
    # 训练/优化
    batch_size = 128             # 🚀 Increased for A10 (24GB VRAM) utilization
    learning_rate = 0.0003       # 保持 3e-4 (Optimizer Safety for V2 Weights)
    weight_decay = 1e-4
    epochs = 120
    
    # 训练策略
    freeze_backbone_epochs = 5
    
    # 数据增强与正则化
    dropout = 0.2
    use_mixup = True             # ✅ Re-enabled: Essential for Manifold Smoothing & Generalization
    
    # ✅ [Added] Random Erasing as Compensation
    use_random_erasing = True
    re_prob = 0.2
    
    mixup_alpha = 0.2
    mixup_prob = 0.5
    
    use_ema = True
    ema_decay = 0.999            # 🛡️ 以 EMA 为准
    
    # 标签平滑 (Label Smoothing)
    label_smoothing = 0.0        # 禁用，避免污染 DLDL 分布
    
    # 数据集开关 (Set use_aaf=False for pure academic benchmark)
    use_afad = True
    use_aaf = False   # 默认开启以增强鲁棒性，若需对比 SOTA 请设为 False

    # 数据集路径 relative to ROOT_DIR
    afad_dir = os.path.join(ROOT_DIR, "datasets", "AFAD")
    aaf_dir = os.path.join(ROOT_DIR, "datasets", "AAF")
    
    # LDS (标签分布平滑)
    use_reweighting = True
    use_alignment = False
    
    lds_sigma = 3                # 📉 窗口收紧至 3
    
    # 图片参数
    img_size = 224
    num_workers = 6              # 🏎️ Optimized for CPU usage (avoid 100% load)
    early_stopping_patience = 999

    def __init__(self):
        pass # Attributes are class-level or properties
        
    def __repr__(self):
        return f"🚀 Starting Project: {self.project_name} on {self.device}"
