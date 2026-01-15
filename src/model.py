
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


# ==========================================
# Coordinate Attention (CA) Module
# ==========================================
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 1. Coordinate Encoding
        # Pool X: (B, C, H, 1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # Pool Y: (B, C, 1, W)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2) # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2) # (B, C, 1, W)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out

# ==========================================
# 🧱 [Innovation] Bottleneck SPP (v2)
# ==========================================
class BottleneckSPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Global-Local Feature Aggregation:
        # Multiscale receptive fields to capture both wrinkles (Local) 
        # and facial structure (Global).
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        
        # Channel Expansion: in_c * 4
        # Channel Compression: -> out_channels
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish()
        )

    def forward(self, x):
        p1 = x
        p2 = self.pool1(x)
        p3 = self.pool2(x)
        p4 = self.pool3(x)
        out = torch.cat([p1, p2, p3, p4], dim=1)
        return self.project(out)

class LightweightAgeEstimator(nn.Module):
    def __init__(self, config):
        super(LightweightAgeEstimator, self).__init__()
        
        self.config = config
        num_classes = config.num_classes
        dropout = getattr(config, 'dropout', 0.2)

        # 骨干网络：MobileNetV3 Large
        # 使用预训练权重加速收敛
        # Upgraded to V2 (New recipe, higher acc: 75.2% vs 74.0%)
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.backbone = mobilenet_v3_large(weights=weights)

        # -----------------------------------------------------------
        # 🌟 Hybrid Attention Innovation 🌟
        # Replace SE-Block with Coordinate Attention (CA) in the last 4 blocks
        # Strategy: "Pyramid Attention Injection"
        # 
        # Rationale: 
        # Shallow layers (Edge/Texture) -> Keep SE or Vanilla (Fast)
        # Deep layers (Semantics) -> Use CA (Shape/Position aware)
        # -----------------------------------------------------------
        
        # In torchvision MobileNetV3:
        # self.backbone.features is a Sequential of InvertedResidual blocks.
        # 🌟 [Innovation] Pyramid Attention Injection
        # Replace the last 4 SE blocks with Coordinate Attention (CA)
        # But ONLY if use_hybrid_attention is True
        use_ca = getattr(self.config, 'use_hybrid_attention', True)
        
        if use_ca:
            print("🚀 [Model] Hybrid Attention Strategy: ENABLED (Injecting CoordAtt)")
            
            # 1. Find blocks with SE
            se_blocks_indices = []
            for i, block in enumerate(self.backbone.features):
                # Search inside block.block (Sequential) for SqueezeExcitation
                if hasattr(block, 'block') and isinstance(block.block, nn.Sequential):
                    for m in block.block:
                        if "SqueezeExcitation" in str(type(m)):
                            se_blocks_indices.append(i)
                            break
            
            # 2. Target the last 4
            target_indices = se_blocks_indices[-4:]
            
            # 3. Replace
            for idx in target_indices:
                block = self.backbone.features[idx]
                se_idx = -1
                original_se = None
                
                # Find the SE layer index again
                for i, module in enumerate(block.block):
                    if "SqueezeExcitation" in str(type(module)):
                        original_se = module
                        se_idx = i
                        break
                
                if original_se is not None:
                    # Get channels (input to SE)
                    # SqueezeExcitation has .fc1 (Conv2d)
                    if hasattr(original_se, 'fc1'):
                        c = original_se.fc1.in_channels
                    elif hasattr(original_se, 'avgpool'): # Some variants
                         # Fallback / Inspection
                         pass # Should be fine with standard torchvision
                         c = original_se.fc1.in_channels # Assume standard
                    else:
                        # Fallback if structure is different
                        continue

                    # Create CA with reduction=16 (commonly used in paper)
                    # SE uses reduction 4 usually in MBV3, but CA paper suggests 32. 
                    # We use 16 as a balance.
                    new_ca = CoordAtt(c, c, reduction=16)
                    
                    # Replace
                    block.block[se_idx] = new_ca
                    # print(f"  -> Replaced SE at Block {idx} Layer {se_idx} with CoordAtt")
        else:
             print("⚠️ [Model] Hybrid Attention Strategy: DISABLED (Using Vanilla MobileNetV3)")

        # 修改分类头 (Classifier)
        # 获取最后一个卷积层输出的特征通道数
        self.last_channel = self.backbone.classifier[0].in_features # 960 or 1280
        
        # 🌟 [Innovation] Multi-Scale Feature Fusion (MSFF)
        self.use_multi_scale = getattr(self.config, 'use_multi_scale', False)
        
        if self.use_multi_scale:
            print("🔥 [Model] Multi-Scale Fusion: ENABLED (Dual-Stage: Texture & Structure)")
            # 🌟 Plan D Improvement (Electronics 2024 inspired)
            # Use Block 6 (28x28) for fine texture (wrinkles)
            # Use Block 12 (14x14) for mid-level structure (shape)
            
            # Index Check for MobileNetV3 Large
            self.idx_shallow = 6   # 28x28, 40 ch (approx)
            self.idx_mid = 12      # 14x14, 112 ch
            
            # Projectors to unify channels before fusion
            # We project everything to 64 channels to keep it lightweight
            fusion_dim = 64
            
            # Shallow Projector: 40 -> 64
            # Note: Input channel 40 is an estimate, we might need dynamic detection or trust standard arch
            # MBV3-Large block 6 output is 40 channels.
            self.project_shallow = nn.Sequential(
                nn.Conv2d(40, fusion_dim, 1, bias=False),
                nn.BatchNorm2d(fusion_dim),
                nn.ReLU(inplace=True)
            )
            
            # Mid Projector: 112 -> 64
            self.project_mid = nn.Sequential(
                nn.Conv2d(112, fusion_dim, 1, bias=False),
                nn.BatchNorm2d(fusion_dim),
                nn.ReLU(inplace=True)
            )
            
            # Attention Weighted Fusion (Simple SE-like weight)
            # Weights for [Shallow, Mid]
            self.fusion_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            
            # Final Projector after fusing (64 -> 128 to match old interface)
            self.fusion_out = nn.Sequential(
                nn.Conv2d(fusion_dim, 128, 1, bias=False),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            
            classifier_input_dim = 1280 + 128
        else:
            print("ℹ️ [Model] Multi-Scale Fusion: DISABLED")
            classifier_input_dim = 1280
            
        # 🌟 [Innovation] Spatial Pyramid Pooling (Low Cost, High Return)
        self.use_spp = getattr(self.config, 'use_spp', False)
        if self.use_spp:
            print("🧱 [Model] SPP Strategy: ENABLED (Bottleneck SPP v2 - Global-Local 5/9/13)")
            self.spp_channels = 512
            self.spp_module = BottleneckSPP(self.last_channel, self.spp_channels)
            
            # Classifier Input Dim: 512 (+ 128 if MSFF)
            if self.use_multi_scale:
                classifier_input_dim = self.spp_channels + 128
            else:
                classifier_input_dim = self.spp_channels
        else:
            print("ℹ️ [Model] SPP Strategy: DISABLED (Using Global Average Pooling)")
            # classifier_input_dim remains what was set by MSFF block
            
            # Fallback Projector for Non-SPP mode (960 -> 1280)
            # Ensures dimension compatibility if MSFF expects 1280 or strictly matching previous logic
            self.project_960_1280 = nn.Sequential(nn.Linear(960, 1280), nn.Hardswish())

        # 替换分类器
        # 我们不再使用 backbone.classifier 作为特征提取器的一部分
        # 而是完全接管 Head
        self.backbone.classifier = nn.Identity() 
        
        # 定义我们自己的 Head
        self.final_head = nn.Sequential(
            nn.Linear(classifier_input_dim, 1024), # 融合后降维 to 1024 (Sweet spot)
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # 双流感知流程 + SPP
        
        # 1. 前向传播直到 Fusion Layer (如果 MSFF 开启) 或者直到最后
        # 为了兼容两种模式，我们需要灵活处理
        
        # 提取中间层 (Texture) 和 最后一层 (Deep)
        x_shallow = x
        feat_texture = None
        
        # Capture Plan D Indices
        target_idx_mid = getattr(self, 'idx_mid', 12)
        target_idx_shallow = getattr(self, 'idx_shallow', 6)
        
        feat_shallow = None
        feat_mid = None
        
        if self.use_multi_scale:
             # Part 1: Iterate through backbone
             for i, layer in enumerate(self.backbone.features):
                 x_shallow = layer(x_shallow)
                 
                 # Capture Shallow (Block 6)
                 if i == target_idx_shallow:
                     feat_shallow = x_shallow 
                     
                 # Capture Mid (Block 12)
                 if i == target_idx_mid:
                     feat_mid = x_shallow
                     break 
                     
             # --- Plan D Fusion Logic ---
             # 1. Project Channels to 64
             f_s = self.project_shallow(feat_shallow) 
             f_m = self.project_mid(feat_mid)         
             
             # 2. Downsample Shallow to match Mid (28->14)
             f_s_down = nn.functional.adaptive_avg_pool2d(f_s, (14, 14))
             
             # 3. Attention Weighted Fusion
             w = torch.nn.functional.softmax(self.fusion_weight, dim=0)
             f_fused = w[0] * f_s_down + w[1] * f_m
             
             # 4. Final Projection -> 128
             feat_texture = self.fusion_out(f_fused) # [B, 128]

             # Part 2: Deep Continuation
             x_deep = x_shallow
             for i in range(target_idx_mid + 1, len(self.backbone.features)):
                  x_deep = self.backbone.features[i](x_deep)
        else:
            # Simple Forward
            x_deep = self.backbone.features(x)
            
        # x_deep is now [B, 960, 7, 7]
        
        # --- SPP Logic ---
        if self.use_spp:
            # 1. Bottleneck SPP v2
            x_spp_out = self.spp_module(x_deep) # [B, 512, 7, 7]
            
            # 2. Global Average Pooling (Standard after SPP feature aggregation)
            x_sem = torch.nn.functional.adaptive_avg_pool2d(x_spp_out, (1, 1)).flatten(1) # [B, 512]
        else:
            # Standard GAP + Project
            x_pool = self.backbone.avgpool(x_deep).flatten(1) # [B, 960]
            x_sem = self.project_960_1280(x_pool)

        # --- Fusion ---
        if self.use_multi_scale and feat_texture is not None:
             x_final = torch.cat([x_sem, feat_texture], dim=1)
        else:
             x_final = x_sem
             
        # Result
        logits = self.final_head(x_final)
        return logits
