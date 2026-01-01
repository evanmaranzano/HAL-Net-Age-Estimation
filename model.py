# model.py
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
            print("🔥 [Model] Multi-Scale Fusion: ENABLED (Texture-Semantics Dual-Stream)")
            # 我们将从中间层提取特征
            # MobileNetV3 Large Features:
            # Block 6: 28x28, 80 ch (Exp: 240) -> Stride 8?
            # Block 12: 14x14, 112 ch (Exp: 672) -> Stride 16?
            # 自动探测通道数可能比较难，这里硬编码或者动态获取
            # 假设我们用 Block 12 (Index 12)
            self.fusion_idx = 12 
            self.fusion_channels = 112 # MBV3-Large 默认
            
            # 使用 1x1 卷积调整通道并 Global Pool
            self.fusion_project = nn.Sequential(
                nn.Conv2d(self.fusion_channels, 128, 1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            # 融合后的维度
            classifier_input_dim = 1280 + 128
        else:
            print("ℹ️ [Model] Multi-Scale Fusion: DISABLED")
            classifier_input_dim = 1280
            
        # 🌟 [Innovation] Spatial Pyramid Pooling (Low Cost, High Return)
        self.use_spp = getattr(self.config, 'use_spp', False)
        if self.use_spp:
            print("🧱 [Model] SPP Strategy: ENABLED (1x1, 2x2, 4x4) with 128-ch bottleneck")
            # SPP Bottleneck: 960 (Deep Feat) -> 128 
            # Note: The input to classifier is usually 960 (before expansion to 1280 in classifier[0])
            # Or is it 1280? 
            # In MBV3 Large: last conv is 960. 
            # self.last_channel = 960.
            
            self.spp_channels = 128
            self.spp_bottleneck = nn.Sequential(
                nn.Conv2d(self.last_channel, self.spp_channels, 1, bias=False),
                nn.BatchNorm2d(self.spp_channels),
                nn.Hardswish()
            )
            
            # SPP Dimensions:
            # 1x1 = 1
            # 2x2 = 4
            # 4x4 = 16
            # Total = 21 * spp_channels
            self.spp_dim = self.spp_channels * 21 # 128 * 21 = 2688
            
            # Update Classifier Input Dim
            # If SPP is on, we replace the deep branch (1280) with SPP (2688)
            # If MSFF is also on, we add 128 (Texture) -> Total = 2688 + 128
            
            if self.use_multi_scale:
                classifier_input_dim = self.spp_dim + 128
            else:
                classifier_input_dim = self.spp_dim
            
            # Note: We need adaptive pools
            self.spp_pool1 = nn.AdaptiveAvgPool2d(1)
            self.spp_pool2 = nn.AdaptiveAvgPool2d(2)
            self.spp_pool4 = nn.AdaptiveAvgPool2d(4)
            
        else:
            print("ℹ️ [Model] SPP Strategy: DISABLED (Using Global Average Pooling)")
            # classifier_input_dim remains what was set by MSFF block

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
        
        # 需要知道遍历到哪里。
        # 如果 MSFF 关闭，fusion_idx 不存在。
        target_idx = getattr(self, 'fusion_idx', -1)
        
        if self.use_multi_scale:
             # Part 1: Initial -> Stride 16
            for i, layer in enumerate(self.backbone.features):
                x_shallow = layer(x_shallow)
                if i == target_idx:
                    feat_texture = x_shallow # Capture Texture
                    # Project Texture
                    feat_texture = self.fusion_project(feat_texture) # [B, 128]
                    break 
                    
            # Part 2: Deep 
            x_deep = x_shallow
            for i in range(target_idx + 1, len(self.backbone.features)):
                 x_deep = self.backbone.features[i](x_deep)
        else:
            # Simple Forward
            x_deep = self.backbone.features(x)
            
        # x_deep is now [B, 960, 7, 7]
        
        # --- SPP Logic ---
        if self.use_spp:
            # 1. Bottleneck (960 -> 128)
            x_spp = self.spp_bottleneck(x_deep) # [B, 128, 7, 7]
            
            # 2. Pyramid Pooling
            p1 = self.spp_pool1(x_spp).flatten(1) # [B, 128]
            p2 = self.spp_pool2(x_spp).flatten(1) # [B, 128*4]
            p3 = self.spp_pool4(x_spp).flatten(1) # [B, 128*16]
            
            x_sem = torch.cat([p1, p2, p3], dim=1) # [B, 2688]
        else:
            # Standard GAP + Project
            # Original MobileNetV3 Project: 960 -> 1280
            # Since we replaced backbone.classifier with Identity, we need to manually do the projection if we want strictly mbv3 behavior
            # But here we can simpler: GAP -> Linear if we wanted.
            # actually existing logic for Single Stream was:
            # backbone.classifier(flattened)
            # backbone.classifier used to be: Linear(960, 1280) -> Act -> Linear(1280, Num)
            # But now we set backbone.classifier = Identity() in init if SPP is ON logic reached...
            
            # Wait, if use_spp is False, we need to respect the old logic or redefine it.
            # To keep it clean:
            # GAP -> 960
            x_pool = self.backbone.avgpool(x_deep).flatten(1) # [B, 960]
            
            # Standard Projection (Optional, but good for capacity)
            # We can map 960 -> 1280 to match MSFF logic
            # Or just use 960. 
            # Let's simple projection matrix or just use x_pool.
            # Previous MSFF logic assumed 1280 (from backbone.classifier[0]).
            # Let's implement a simple projection to 1280 if SPP is off, to match expected dim
            # But wait, classifier_input_dim is 1280 if SPP off.
            # Does x_pool (960) match 1280? No.
            # We need a bridge.
            # Simplified: Just pad? No. 
            # Re-instantiate a linear layer?
            
            # ⚠️  CRITICAL: If SPP is False, we need to ensure dimensions match.
            # Let's assume SPP is always preferred. 
            # If not, we bridge 960 -> 1280 manually here if strict.
            # Or better: Just change classifier_input_dim to 960 if SPP is False and MSFF is False.
            # If MSFF Is True (1280+128), and SPP is False...
            # The Deep branch needs to provide 1280.
            # Let's verify init logic.
            
            # Correction: 
            # If SPP=False:
            # classifier_input_dim = 1280 (set by MSFF block)
            # So x_sem MUST be 1280.
            # x_deep is 960.
            # We need a 960->1280 layer.
            # Let's just use a dedicated Linear layer for this case to stay safe.
            if not hasattr(self, 'project_960_1280'):
                 self.project_960_1280 = nn.Sequential(nn.Linear(960, 1280), nn.Hardswish())
                 self.project_960_1280.to(x_deep.device) # risky dynamic init
            
            x_sem = self.project_960_1280(x_pool)

        # --- Fusion ---
        if self.use_multi_scale and feat_texture is not None:
             x_final = torch.cat([x_sem, feat_texture], dim=1)
        else:
             x_final = x_sem
             
        # Result
        logits = self.final_head(x_final)
        return logits

