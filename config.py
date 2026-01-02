import torch

class Config:
    def __init__(self):
        # 基础名称
        base_name = "FADE-Net"
        
        # 动态后缀生成
        suffixes = []
        if getattr(self, 'use_hybrid_attention', True):
            suffixes.append("HA")
        if getattr(self, 'use_dldl_v2', True):
            suffixes.append("DLDL")
        if getattr(self, 'use_multi_scale', True):
            suffixes.append("MSFF")
        if getattr(self, 'use_spp', True):
            suffixes.append("SPP")
            
        if not suffixes:
            suffixes.append("Baseline")
            
        # 组合最终名称
        self.project_name = f"{base_name}_{'_'.join(suffixes)}"
        
    # 基础配置 (Class Attributes to be overridden by instance attributes if needed, 
    # but since we use cfg = Config(), we can access instance attrs)
    
    # ⚠️ 注意: 下面的属性是类属性。在 __init__ 中我们定义了实例属性 project_name。
    # Python 实例访问属性时，如果实例字典里有，就优先用实例的。
    # 所以这没问题。
    
    # project_name = "HAL-Net_Age_Estimation" # 移至 __init__ 动态生成
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径 (已移除 UTKFace)
    # train_dir = "./data_aligned/UTKFace/train" 
    # val_dir = "./data_aligned/UTKFace/val"
    afad_dir = "./data_aligned/AFAD"
    aaf_dir = "./data_aligned/AAF"

    # 图片参数
    img_size = 224  

    # 🎯 [修正] 聚焦 0-80 岁黄金区间
    min_age = 0
    max_age = 80         # 适配 AFAD (15-75) 和 AAF 主分布
    num_classes = 81     # 0-80岁
    sigma = 2.0          

    # 训练参数
    batch_size = 64
    learning_rate = 0.0003
    epochs = 120  
    
    # 优化参数
    weight_decay = 1e-4  
    num_workers = 4
    
    dropout = 0.2              
    early_stopping_patience = 999  
    
    # 数据增强
    use_mixup = True           
    mixup_alpha = 0.2          
    mixup_prob = 0.5           
    
    # DLDL & 采样策略
    use_adaptive_sigma = True  
    sigma_min = 0.8 
    sigma_max = 3.5 
    label_smoothing = 0.0 

    # 训练策略 (Freeze)
    freeze_backbone_epochs = 5 # 前5个Epoch冻结骨干网络，只训练CA层和Head


    use_alignment = False      
    
    lambda_l1 = 0.1            
    lambda_rank = 0.45          
    
    use_reweighting = True     # 需要 LDS 解决 25岁 vs 80岁 不平衡
    
    # EMA
    use_ema = True             
    ema_decay = 0.999

    # 🔬 Ablation Switch (消融实验开关)
    use_hybrid_attention = True  # True=HAL-Net (CA), False=Baseline (SE)
    use_dldl_v2 = True           # True=Adaptive Sigma + Rank Loss, False=Standard DLDL
    use_multi_scale = True       # True=Feature Fusion (Texture Boost), False=Single Stream
    use_spp = True               # True=Spatial Pyramid Pooling (1x1, 2x2, 4x4), False=GAP
