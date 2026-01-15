"""
Advanced TTA and Multi-Seed Ensemble Evaluation

使用方法:
    python scripts/advanced_eval.py [--seed SEED] [--tta MODE] [--ensemble]

功能:
    1. 扩展 TTA (Multi-Scale 3尺度 + flip = 6x 平均)
    2. 多种子 Ensemble
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from config import Config
from model import LightweightAgeEstimator
from dataset import get_dataloaders
from torchvision import transforms
from PIL import Image


def five_crop_tta(images, model, device):
    """
    Apply multi-scale TTA: 3 scales (0.9, 1.0, 1.1) x 2 (original + flip) = 6x average.
    Returns averaged probability distribution.
    """
    B, C, H, W = images.shape
    
    # Multi-scale: 0.9, 1.0, 1.1
    scales = [0.9, 1.0, 1.1]
    all_probs = []
    
    for scale in scales:
        if scale != 1.0:
            new_size = int(224 * scale)
            resized = F.interpolate(images, size=new_size, mode='bilinear', align_corners=False)
            # Center crop back to 224
            if new_size > 224:
                start = (new_size - 224) // 2
                resized = resized[:, :, start:start+224, start:start+224]
            else:
                # Pad to 224
                pad = (224 - new_size) // 2
                resized = F.pad(resized, (pad, 224-new_size-pad, pad, 224-new_size-pad), mode='reflect')
        else:
            resized = images
        
        # Original
        logits = model(resized)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)
        
        # Horizontal Flip
        flipped = torch.flip(resized, dims=[3])
        logits_flip = model(flipped)
        probs_flip = F.softmax(logits_flip, dim=1)
        all_probs.append(probs_flip)
    
    # Average all (6x)
    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    return avg_probs


def evaluate_with_tta(model, test_loader, config, device, tta_mode='flip'):
    """
    Evaluate model with TTA.
    
    tta_mode:
        'none': No TTA
        'flip': Horizontal flip only (2x)
        'multi': Multi-scale + flip (6x)
    """
    model.eval()
    test_mae = 0.0
    count = 0
    rank_arange = torch.arange(config.num_classes, device=device).float()
    
    with torch.no_grad():
        for images, labels, ages in tqdm(test_loader, desc=f"Eval ({tta_mode})"):
            images = images.to(device)
            ages = ages.to(device)
            
            if tta_mode == 'none':
                logits = model(images)
                probs = F.softmax(logits, dim=1)
            elif tta_mode == 'flip':
                # Standard: Original + Flip
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = F.softmax(logits_flip, dim=1)
                
                probs = (probs + probs_flip) / 2.0
            else:  # 'multi'
                probs = five_crop_tta(images, model, device)
            
            output_ages = torch.sum(probs * rank_arange, dim=1)
            mae = torch.abs(output_ages - ages).sum().item()
            test_mae += mae
            count += images.size(0)
    
    return test_mae / count


def ensemble_predict(models, images, device):
    """
    Ensemble prediction: average probability distributions from multiple models.
    """
    all_probs = []
    
    for model in models:
        # Original
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        
        # Flip
        images_flip = torch.flip(images, dims=[3])
        logits_flip = model(images_flip)
        probs_flip = F.softmax(logits_flip, dim=1)
        
        avg = (probs + probs_flip) / 2.0
        all_probs.append(avg)
    
    # Average across models
    return torch.stack(all_probs, dim=0).mean(dim=0)


def main():
    parser = argparse.ArgumentParser(description="Advanced Evaluation")
    parser.add_argument('--seed', type=int, default=1337, help='Seed to evaluate (default: 1337)')
    parser.add_argument('--tta', type=str, default='flip', choices=['none', 'flip', 'multi'], help='TTA mode')
    parser.add_argument('--ensemble', action='store_true', help='Use multi-seed ensemble')
    parser.add_argument('--seeds', type=str, default='42,1337', help='Seeds for ensemble (comma-separated)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config()
    
    # Load data
    print("📊 Loading test data...")
    _, _, test_loader, _ = get_dataloaders(cfg)
    rank_arange = torch.arange(cfg.num_classes, device=device).float()
    
    if args.ensemble:
        # Multi-seed ensemble
        seeds = [int(s) for s in args.seeds.split(',')]
        print(f"🎯 Ensemble Mode: Seeds {seeds}")
        
        models = []
        for seed in seeds:
            model_path = os.path.join(ROOT_DIR, f'best_model_FADE-Net_HA_DLDL_MSFF_SPP_MV_seed{seed}.pth')
            if not os.path.exists(model_path):
                print(f"⚠️ Model not found: {model_path}")
                continue
            
            model = LightweightAgeEstimator(cfg)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
            print(f"✅ Loaded: {os.path.basename(model_path)}")
        
        if len(models) < 2:
            print("❌ Need at least 2 models for ensemble!")
            return
        
        # Evaluate ensemble
        test_mae = 0.0
        count = 0
        
        with torch.no_grad():
            for images, labels, ages in tqdm(test_loader, desc="Ensemble Eval"):
                images = images.to(device)
                ages = ages.to(device)
                
                probs = ensemble_predict(models, images, device)
                output_ages = torch.sum(probs * rank_arange, dim=1)
                mae = torch.abs(output_ages - ages).sum().item()
                test_mae += mae
                count += images.size(0)
        
        final_mae = test_mae / count
        print(f"\n🏆 Ensemble Test MAE (Seeds {seeds}): {final_mae:.4f}")
        
    else:
        # Single model evaluation with TTA
        model_path = os.path.join(ROOT_DIR, f'best_model_FADE-Net_HA_DLDL_MSFF_SPP_MV_seed{args.seed}.pth')
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        model = LightweightAgeEstimator(cfg)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print(f"✅ Loaded: {os.path.basename(model_path)}")
        
        # Evaluate with different TTA modes
        print(f"\n📊 Evaluating with TTA mode: {args.tta}")
        
        test_mae = evaluate_with_tta(model, test_loader, cfg, device, args.tta)
        print(f"\n🏆 Test MAE (Seed {args.seed}, TTA={args.tta}): {test_mae:.4f}")


if __name__ == '__main__':
    main()
