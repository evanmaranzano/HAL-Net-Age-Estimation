"""
SWA (Stochastic Weight Averaging) Model Generator

使用方法:
    python scripts/swa_average.py [--seed SEED] [--eval]

功能:
    1. 平均最后 10 个 epoch 的 checkpoint 生成 SWA 模型
    2. 可选：直接评估 SWA 模型性能
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from config import Config
from model import LightweightAgeEstimator
from dataset import get_dataloaders


def average_checkpoints(checkpoint_paths, device='cpu'):
    """
    Average model weights from multiple checkpoints.
    """
    print(f"📊 Averaging {len(checkpoint_paths)} checkpoints...")
    
    avg_state = None
    n = len(checkpoint_paths)
    
    for i, path in enumerate(checkpoint_paths):
        print(f"   Loading [{i+1}/{n}]: {os.path.basename(path)}")
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state_dict.items()}
        else:
            for k in avg_state.keys():
                avg_state[k] += state_dict[k].float()
    
    # Divide by number of checkpoints
    for k in avg_state.keys():
        avg_state[k] /= n
    
    return avg_state


def evaluate_model(model, test_loader, config, device):
    """
    Evaluate model on test set with TTA.
    """
    model.eval()
    test_mae = 0.0
    count = 0
    rank_arange = torch.arange(config.num_classes).to(device)
    
    with torch.no_grad():
        for images, labels, ages in test_loader:
            images = images.to(device)
            ages = ages.to(device)
            
            # TTA: Original + Horizontal Flip
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = F.softmax(logits_flip, dim=1)
            
            probs = (probs + probs_flip) / 2.0
            
            output_ages = torch.sum(probs * rank_arange.float(), dim=1)
            mae = torch.abs(output_ages - ages).sum().item()
            test_mae += mae
            count += images.size(0)
    
    return test_mae / count


def main():
    parser = argparse.ArgumentParser(description="SWA Model Generator")
    parser.add_argument('--seed', type=int, default=None, help='Seed to process (default: all available)')
    parser.add_argument('--eval', action='store_true', help='Evaluate SWA model after generation')
    parser.add_argument('--epochs', type=str, default='111-120', help='Epoch range to average (default: 111-120)')
    args = parser.parse_args()
    
    # Determine seeds to process
    if args.seed:
        seeds = [args.seed]
    else:
        # Auto-detect available seeds
        seeds = []
        for s in [42, 1337, 3407]:
            if os.path.exists(os.path.join(ROOT_DIR, f'checkpoint_seed{s}_epoch_111.pth')):
                seeds.append(s)
        print(f"🔍 Auto-detected seeds: {seeds}")
    
    if not seeds:
        print("❌ No checkpoint files found!")
        return
    
    # Parse epoch range
    start_epoch, end_epoch = map(int, args.epochs.split('-'))
    
    results = {}
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"🌱 Processing Seed {seed}")
        print(f"{'='*60}")
        
        # Collect checkpoint paths
        checkpoint_paths = []
        for epoch in range(start_epoch, end_epoch + 1):
            path = os.path.join(ROOT_DIR, f'checkpoint_seed{seed}_epoch_{epoch}.pth')
            if os.path.exists(path):
                checkpoint_paths.append(path)
            else:
                print(f"⚠️ Missing: {path}")
        
        if len(checkpoint_paths) < 5:
            print(f"⚠️ Not enough checkpoints for Seed {seed} (found {len(checkpoint_paths)})")
            continue
        
        # Average checkpoints
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        avg_state = average_checkpoints(checkpoint_paths, device)
        
        # Save SWA model
        swa_path = os.path.join(ROOT_DIR, f'swa_model_seed{seed}.pth')
        torch.save(avg_state, swa_path)
        print(f"✅ SWA model saved: {swa_path}")
        
        # Evaluate if requested
        if args.eval:
            print(f"\n📊 Evaluating SWA model...")
            
            cfg = Config()
            model = LightweightAgeEstimator(cfg)
            model.load_state_dict(avg_state)
            model.to(device)
            
            _, _, test_loader, _ = get_dataloaders(cfg)
            
            test_mae = evaluate_model(model, test_loader, cfg, device)
            print(f"🏆 SWA Test MAE (Seed {seed}): {test_mae:.4f}")
            results[seed] = test_mae
            
            # Compare with original best model
            best_model_path = os.path.join(ROOT_DIR, f'best_model_FADE-Net_HA_DLDL_MSFF_SPP_MV_seed{seed}.pth')
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                orig_mae = evaluate_model(model, test_loader, cfg, device)
                print(f"📋 Original Best MAE: {orig_mae:.4f}")
                print(f"📈 Improvement: {orig_mae - test_mae:+.4f}")
    
    if results:
        print(f"\n{'='*60}")
        print("📊 Summary")
        print(f"{'='*60}")
        for s, mae in results.items():
            print(f"  Seed {s}: {mae:.4f}")


if __name__ == '__main__':
    main()
