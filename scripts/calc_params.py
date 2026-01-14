import torch
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model import LightweightAgeEstimator
from thop import profile, clever_format

def main():
    cfg = Config()
    model = LightweightAgeEstimator(cfg)
    
    # 1. Basic Count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Total Params (M): {total_params / 1e6:.2f}M")

    # 2. THOP (FLOPs)
    try:
        input = torch.randn(1, 3, cfg.img_size, cfg.img_size)
        flops, params = profile(model, inputs=(input, ), verbose=False)
        flops_fmt, params_fmt = clever_format([flops, params], "%.2f")
        print(f"FLOPs: {flops_fmt}")
        print(f"Params (THOP): {params_fmt}")
    except Exception as e:
        print(f"THOP calculation failed (optional): {e}")

if __name__ == "__main__":
    main()
