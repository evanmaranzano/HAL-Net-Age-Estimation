# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
# import mediapipe as mp
import copy
import random
import os

# ==========================================
# 0. Reproducibility Tool
# ==========================================
def seed_everything(seed=42):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Global Seed Set to {seed}")

# ==========================================
# 1. DLDL 核心处理类
# ==========================================
class DLDLProcessor:
    def __init__(self, config):
        self.max_age = config.max_age
        self.num_classes = config.num_classes
        
        # 动态 sigma 参数
        self.use_dldl_v2 = getattr(config, 'use_dldl_v2', True)
        
        # Original logic: use_adaptive_sigma was a separate flag.
        # Now we merge it into 'use_dldl_v2' for ablation simplicity,
        # OR we keep it independent but default it to True if dldl_v2 is True.
        
        # Let's say: use_adaptive_sigma is active ONLY IF use_dldl_v2 is True.
        self.use_adaptive_sigma = self.use_dldl_v2 and getattr(config, 'use_adaptive_sigma', False)
        
        if self.use_adaptive_sigma:
            self.sigma_min = getattr(config, 'sigma_min', 1.5)
            self.sigma_max = getattr(config, 'sigma_max', 3.5)
            # print("✅ [Loss] Adaptive Sigma: ENABLED") # avoid spamming
        else:
            self.sigma = config.sigma
            # print(f"ℹ️ [Loss] Adaptive Sigma: DISABLED (Fixed sigma={self.sigma})")
        
        # Label Smoothing 参数
        self.label_smoothing = getattr(config, 'label_smoothing', 0.0)
        
        # 预先生成年龄索引张量 [0, 1, ..., num_classes-1]
        self.age_indices = torch.arange(0, config.num_classes, dtype=torch.float32)

    def generate_label_distribution(self, age_scalar, sigma_offset=0.0):
        """
        将标量年龄转化为离散的高斯概率分布。
        改进:
        1. 动态 sigma: 年龄越大,不确定性越高
        2. Label Smoothing: 平滑分布,防止过拟合
        3. Sigma Jitter: 训练时随机扰动 sigma
        """
        # 2024-12-16: Convert numpy scalar to tensor to prevent warning
        if not isinstance(age_scalar, torch.Tensor):
            age_scalar = torch.tensor(age_scalar, dtype=torch.float32)

        # 动态计算 sigma (年龄越大,sigma 越大)
        if self.use_adaptive_sigma:
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * (age_scalar / self.max_age)
        else:
            sigma = self.sigma
            
        # Apply Jitter (add offset)
        sigma = sigma + sigma_offset
        # Ensure sigma doesn't go too low (e.g. < 0.5)
        sigma = max(sigma, 0.5)

        # 计算每个年龄节点 j 与 真实年龄 y 的差异
        # $$P(j|x) \propto e^{-\frac{(j-y)^2}{2\sigma^2}}$$
        diff = self.age_indices - age_scalar
        prob_dist = torch.exp(-0.5 * (diff / sigma) ** 2)

        # 归一化保证概率和为1
        prob_dist = prob_dist / torch.sum(prob_dist)
        
        # Label Smoothing: 混合均匀分布
        if self.label_smoothing > 0:
            uniform_dist = torch.ones_like(prob_dist) / self.num_classes
            prob_dist = (1 - self.label_smoothing) * prob_dist + self.label_smoothing * uniform_dist

        return prob_dist

    def expectation_regression(self, predicted_probs):
        """
        推理端：对预测分布进行加权求和。
        """
        if predicted_probs.device != self.age_indices.device:
            self.age_indices = self.age_indices.to(predicted_probs.device)

        # 期望计算：概率 * 年龄值 的总和
        batch_expected_age = torch.sum(predicted_probs * self.age_indices, dim=1)
        return batch_expected_age

# ==========================================
# 2. 人脸对齐工具 (Face Aligner)
# ==========================================
class FaceAligner:
    def __init__(self):
        import mediapipe as mp
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.3)

    def align(self, image, desired_size=224, desired_left_eye=(0.32, 0.35)):
        """
        标准人脸对齐 (Similarity Transform)
        1. 旋转: 眼睛连线水平
        2. 缩放+平移: 将眼睛固定在图片的特定位置
        Output: 224x224 (默认) 的标准人脸图
        """
        try:
            img_np = np.array(image)
            h, w, c = img_np.shape
            
            # 检测人脸
            results = self.detector.process(img_np)
            
            if not results.detections:
                return None # 检测失败，直接丢弃
            
            # 取置信度最高的一个
            detection = results.detections[0]
            # score = detection.score[0]
            # if score < 0.5: # 再次过滤低置信度 -> 已由 init 参数控制，此处移除以避免矛盾
            #    return None
                
            keypoints = detection.location_data.relative_keypoints
            # MediaPipe: 0=右眼(图左), 1=左眼(图右)
            right_eye = np.array([keypoints[0].x * w, keypoints[0].y * h])
            left_eye = np.array([keypoints[1].x * w, keypoints[1].y * h])
            
            # 1. 计算角度 (使得 左眼-右眼 连线水平)
            # dy = right - left (y轴向下)
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            
            # 2. 计算当前瞳距
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            
            # 3. 计算目标瞳距
            # desired_left_eye[0] 是左眼(图右)的 x 比例？ 
            # 通常 convention: desired_left_eye=(0.35, 0.35) 指的是 "Left Eye" (subject's left, image right) 的位置？
            # 不，通常指的是 "Image Left" 那个眼睛 (即 Right Eye of subject) 在 0.35。
            # 让我们明确定义：
            # 我们希望：Subject's Right Eye (Image Left) at x = 1.0 - desired_left_eye_x ? No.
            # 让我们用标准常用值：
            # Subject's Right Eye (Image Left) -> (desired_dist_x, desired_dist_y)
            # Subject's Left Eye (Image Right) -> (1-desired_dist_x, desired_dist_y)
            
            desired_right_eye_x = 1.0 - desired_left_eye[0] # e.g. 1 - 0.32 = 0.68
            
            # 目标瞳距 (像素)
            desired_dist_pix = (desired_right_eye_x - desired_left_eye[0]) * desired_size
            
            # 4. 计算缩放比例
            scale = desired_dist_pix / dist
            
            # 5. 计算旋转+缩放矩阵
            # 旋转中心: 两眼中心
            eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
            
            # 6. 加入平移分量 (Update Translation)
            # 我们希望 eyes_center 移动到图片中心 (desired_size*0.5, desired_size*desired_eye_y)
            tX = desired_size * 0.5
            tY = desired_size * desired_left_eye[1]
            
            M[0, 2] += (tX - eyes_center[0])
            M[1, 2] += (tY - eyes_center[1])
            
            # 7. Warp
            aligned_np = cv2.warpAffine(img_np, M, (desired_size, desired_size), flags=cv2.INTER_CUBIC)
            
            return Image.fromarray(aligned_np)

        except Exception as e:
            # print(f"Align Fail: {e}")
            return None

# ==========================================
# 3. EMA (指数滑动平均)
# ==========================================
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ==========================================
# 4. 高级损失函数 (Ranking Loss)
# ==========================================
class OrderRegressionLoss(nn.Module):
    """
    排序损失 (Ordinal Regression / Ranking Loss)
    迫使模型学习 '年龄 A > 年龄 B' 这种序关系。
    Ref: "Rank consistent ordinal regression for neural networks with application to age estimation" (CVPR 2016)
    或者简化版: 对 logits 施加约束 or 对 probabilities 施加 Rank Loss
    
    这里采用一种简单且有效的策略:
    Soft Ranking Loss on Expectation (借鉴 Mean-Variance Loss 思想)
    或者对 Logits 进行 Pairwise 约束 (耗时)
    
    考虑到计算效率，我们这里使用:
    Binary Ordinal Classification Logic 的变体
    让模型输出不仅仅是分类，还要满足 Ordinal 约束。
    
    Given logits (N, K), we want:
    if y = 3, then P(y>0)=1, P(y>1)=1, P(y>2)=1, P(y>3)=0...
    """
    def __init__(self, config):
        super(OrderRegressionLoss, self).__init__()
        self.num_classes = config.num_classes
        # Register buffer to avoid device issues
        self.register_buffer('rank_indices', torch.arange(self.num_classes).float().unsqueeze(0)) # [1, K]

    def forward(self, logits, true_ages):
        """
        logits: [B, K]
        true_ages: [B]
        """
        # Ordinal Regression label encoding
        # Example: Age 3, K=5. Label=[1, 1, 1, 0, 0]
        # P(y > k) should be 1 if true_age > k
        
        # true_ages: [B, 1]
        t_ages = true_ages.unsqueeze(1)
        
        # Binary Targets: [B, K]
        # rank_indices is [1, K]
        # target[i, k] = 1 if true_age[i] > k else 0
        targets = (t_ages > self.rank_indices).float()
        
        # 我们希望 Logits 能够反映这种 binary 概率
        # 但 Model 输出的是 Softmax Logits，并不直接对应 Binary Classifiers
        # 这里使用一种 Proxy： Cumulative Sum of Softmax? No.
        
        # 更好的 Ranking Loss (Niu et al. CVPR 2016) 需要模型输出 2K 个值的 Binary Logits
        # 这里我们的模型是标准的 Softmax Multi-class
        # 所以我们改用: "Soft Softmax Ranking" 
        # 惩罚: 如果 P(k) 高，那么 P(k-1) around it 也应该合理，
        # 更直接的是：直接惩罚 Expectation 的 L1 (已经在 CombinedLoss 里了)
        
        # ---- 修正方案 ----
        # 鉴于只修改 Loss 不改模型结构 (Model 输出 output=num_classes)
        # 我们使用 "Expectation Ranking" 无需额外 Loss，L1 已经做了。
        # 这里如果一定要加 Rank Loss，通常是指 Pairwise Ranking
        # 随机抽取 Pairs (i, j)，如果 age_i > age_j，则 expectation_i > expectation_j + margin
        
        preds_age = torch.sum(F.softmax(logits, dim=1) * self.rank_indices, dim=1)
        
        # Pairwise Ranking
        n = preds_age.size(0)
        # 随机采样一些 pairs (为了效率，不全量)
        # 也可以简单的: Shuffle and Compare
        idx = torch.randperm(n).to(logits.device)
        preds_shuffled = preds_age[idx]
        ages_shuffled = true_ages[idx]
        
        # diff truth
        diff_truth = true_ages - ages_shuffled
        # diff pred
        diff_pred = preds_age - preds_shuffled
        
        # 如果 truthA > truthB (diff > 0), hope predA > predB (diff > 0)
        # Loss = ReLU( - sign(diff_truth) * diff_pred ) ?
        # 简单点: Sign Consistency
        # Loss = max(0, - sign(diff_truth) * diff_pred + margin)? 
        # No, for regression, L1 is optimal. 
        
        # 回归到原论文思想 (DLDL + Ranking?)
        # 这里的 Rank Loss 如果是 "Auxiliary"，通常是指 Dr. DLDL 提到的
        # "K-1 Binary Classifiers" (需要修改模型最后一层)
        
        # 既然我们不想改模型结构，我们保留这个 Placeholder 为 L1 Loss 的增强版
        # 或者使用 "Distribution Cumulative Loss" -> CDF Loss
        # Minimize KL between CDFs (Cumulative Distribution Functions)
        # Earth Mover's Distance (EMD) 近似
        
        # CDF Calculation
        probs = F.softmax(logits, dim=1)
        cdf_pred = torch.cumsum(probs, dim=1)
        
        # True CDF (Target Distribution 的 CDF)
        # 我们可以直接用 Heaviside Step Function on True Age?
        # 或者 DLDL Target 的 CDF
        # 简单起见，用 True Age 的 Heaviside
        cdf_target = utils_cdf(true_ages, self.num_classes, logits.device)
        
        # EMD Loss (approx by L1 of CDFs)
        loss_emd = F.mse_loss(cdf_pred, cdf_target) # MSE on CDF is stricter
        return loss_emd

def utils_cdf(age, num_classes, device):
    # helper: create heaviside CDF
    # age [B], output [B, K]
    indices = torch.arange(num_classes, device=device).unsqueeze(0)
    # CDF: 1 if idx < age, else 0? No.
    # CDF(k) = P(X <= k)
    # if true_age = 3.5. 
    # k=0 (<=3.5? Yes), k=1(Yes)... k=3(Yes), k=4(No)
    # So 1 if k < true_age ? 
    # Typically CDF is 1 for k >= true_age.
    # Let takes floor.
    mask = (indices >= age.unsqueeze(1)).float() 
    # target CDF: 0 0 0 1 1 1 ... (step at age)
    # Actually: 0 0 0 ... until age, then 1.
    return (indices >= age.unsqueeze(1)).float()

class CombinedLoss(nn.Module):
    def __init__(self, config, weights=None):
        super(CombinedLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.lambda_l1 = getattr(config, 'lambda_l1', 0.1)
        self.lambda_rank = getattr(config, 'lambda_rank', 0.5) # 获取 rank weight
        
        self.dldl = DLDLProcessor(config)
        self.weights = weights 
        
        # 使用 CDF loss 作为 "Rank/Structure" Loss
        # Only init if use_dldl_v2 is True
        self.use_dldl_v2 = getattr(config, 'use_dldl_v2', True)
        if self.use_dldl_v2:
            self.rank_loss_fn = OrderRegressionLoss(config)
            # print("✅ [Loss] Ranking Loss: ENABLED")
        else:
            self.rank_loss_fn = None
            # print("ℹ️ [Loss] Ranking Loss: DISABLED (Standard L1+KL)")

    def forward(self, log_probs, target_dists, true_ages, logits):
        # 1. KL 散度 (Main Loss)
        kl = self.kl_loss(log_probs, target_dists)
        
        # 2. Re-weighting (LDS)
        if self.weights is not None:
            element_kl = F.kl_div(log_probs, target_dists, reduction='none').sum(dim=1)
            age_indices = true_ages.long().clamp(0, len(self.weights)-1)
            batch_weights = self.weights[age_indices]
            w_kl = (element_kl * batch_weights).mean()
        else:
            w_kl = kl
            
        # 3. L1 Loss (Auxiliary)
        probs = torch.exp(log_probs)
        pred_age = self.dldl.expectation_regression(probs)
        
        if self.weights is not None:
             # L1 Loss should also be re-weighted to focus on rare ages
             l1_element = F.l1_loss(pred_age, true_ages, reduction='none')
             l1 = (l1_element * batch_weights).mean()
        else:
             l1 = F.l1_loss(pred_age, true_ages)
        
        # 4. Rank Loss (CDF Loss / EMD)
        # 注意: OrderRegressionLoss 内部实现了 CDF MSE
        if self.use_dldl_v2 and self.rank_loss_fn is not None:
            rank_loss = self.rank_loss_fn(logits, true_ages)
            term_rank = self.lambda_rank * rank_loss
        else:
            rank_loss = torch.tensor(0.0).to(log_probs.device)
            term_rank = 0.0
        
        # 总损失
        total_loss = w_kl + self.lambda_l1 * l1 + term_rank
        return total_loss, w_kl.item(), l1.item(), rank_loss.item()
