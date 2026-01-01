import torch

class Config:
    # 基础配置
    project_name = "MobileNetV3_Age_DLDL_AFAD_AAF_Nodes"
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
    learning_rate = 0.001
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
    sigma_min = 1.0 
    sigma_max = 3.5 
    label_smoothing = 0.05 

    # 训练策略 (Freeze)
    freeze_backbone_epochs = 5 # 前5个Epoch冻结骨干网络，只训练CA层和Head


    use_alignment = False      
    
    lambda_l1 = 0.1            
    lambda_rank = 0.5          
    
    use_reweighting = True     # 需要 LDS 解决 25岁 vs 80岁 不平衡
    
    # EMA
    use_ema = True             
    ema_decay = 0.999
