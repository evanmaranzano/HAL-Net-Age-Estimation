# model.py
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
    def __init__(self, num_classes=101, dropout=0.2):
        super(LightweightAgeEstimator, self).__init__()

        # 骨干网络：MobileNetV3 Large
        # 使用预训练权重加速收敛
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
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
        # We need to find the blocks that have SE.
        # Typically blocks 13, 14, 15, 16 are deep.
        
        target_blocks_count = 0
        target_indices = []
        
        # Reverse search to find the last 4 SE blocks
        for i in range(len(self.backbone.features) - 1, -1, -1):
            block = self.backbone.features[i]
            # Check if it is an InvertedResidual and has 'se' module
            if hasattr(block, 'block') and hasattr(block.block, 'se'):
                 # Check if SE is not Identity (some implementations use Identity for 'no se')
                if isinstance(block.block.se, nn.Module) and not isinstance(block.block.se, nn.Identity):
                    target_indices.append(i)
                    target_blocks_count += 1
                    if target_blocks_count >= 4:
                        break
        
        print(f"🔧 Injecting Coordinate Attention into blocks: {target_indices}")
        
        # Do the replacement
        for i in target_indices:
            block = self.backbone.features[i]
            # Get channels
            # Usually input to SE is the expanded channels
            # Look at torchvision verification: 
            # InvertedResidual(inp, oup, ..) -> block=[Scan, SE, Point]
            # se input dim is typically 'expanded_channels'
            
            # Hack: inspect the existing SE linear layer to find channels
            # torchvision SE: AvgPool -> FC1 -> ReLU -> FC2 -> Sigmoid
            # But torchvision implementation: SqueezeExcitation(input_channels, squeeze_channels)
            # We can trust 'block.block.se.fc1.in_channels' if it exists, 
            # Or usually 'block.block[1]' is SE? No, block structure varies.
            
            # Better way: MobileNetV3 InvertedResidual structure in Torchvision:
            # self.block = nn.Sequential(conv, bn, act, depth_conv, bn, act, se, point_conv, bn, act) ?
            # Wait, standard InvertedResidual:
            # Expand(1x1) -> Depthwise(3x3) -> SE -> Project(1x1)
            # Torchvision implementation uses:
            # layers.append(Conv2dNormActivation(...)) # Expand
            # layers.append(Conv2dNormActivation(...)) # Depthwise
            # layers.append(SqueezeExcitation(...))    # SE
            # layers.append(Conv2dNormActivation(...)) # Project
            
            # So block.block is a nn.Sequential. We search for SqueezeExcitation inside it.
            
            original_se = None
            se_idx = -1
            
            for idx, module in enumerate(block.block):
                if "SqueezeExcitation" in str(type(module)):
                    original_se = module
                    se_idx = idx
                    break
            
            if original_se is not None:
                # Get channels from original SE
                # SqueezeExcitation(input_channels, squeeze_channels)
                # module.fc1 is Conv2d
                c = original_se.fc1.in_channels
                
                # Create CA
                # Reduction can be calibrated. SE uses 4. CA paper uses 32 usually.
                # Let's use a safe dynamic reduction to keep parameter count similar or slightly higher.
                new_ca = CoordAtt(c, c, reduction=16) 
                
                # Replace
                block.block[se_idx] = new_ca
                # print(f"  -> Replaced SE at block[{i}].layer[{se_idx}] with CA(c={c})")

        # 修改分类头 (Classifier)
        # 原始 MobileNetV3 输出层结构较大，这里保留轻量化特性
        # 获取最后一个卷积层输出的特征通道数
        in_features = self.backbone.classifier[0].in_features

        # 替换分类器
        # 本研究修改输出层为全连接层(节点数对应年龄范围) [cite: 15]
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(),
            nn.Dropout(p=dropout),  # Dynamic Dropout from Config
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        # 直接输出 logits
        return self.backbone(x)

