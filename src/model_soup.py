import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
from config import Config
from model import LightweightAgeEstimator
from dataset import get_dataloaders
from utils import DLDLProcessor

def validate(model, loader, device, dldl_tools):
    model.eval()
    mae_sum = 0.0
    count = 0
    rank_arange = torch.arange(81).to(device).float() # Hardcoded for 0-80
    
    with torch.no_grad():
        for images, _, true_ages in loader:
            images = images.to(device)
            true_ages = true_ages.to(device)
            
            # TTA: 1. Normal
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            
            # TTA: 2. Flip
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = F.softmax(logits_flip, dim=1)
            
            # TTA: 3. Fuse
            probs = (probs + probs_flip) / 2.0
            
            # Expectation Regression
            pred_ages = torch.sum(probs * rank_arange, dim=1)
            mae_sum += torch.sum(torch.abs(pred_ages - true_ages)).item()
            count += images.size(0)
            
    return mae_sum / count

def average_weights(checkpoints, include_best=False, best_path="best_model.pth"):
    if include_best and os.path.exists(best_path):
        print(f"🌟 Adding {best_path} to the soup recipe!")
        checkpoints.append(best_path)
        
    if not checkpoints:
        print("❌ No checkpoints to soup.")
        return None

    avg_state_dict = {}
    print(f"🥣 Heating up soup with {len(checkpoints)} ingredients...")
    
    for i, path in enumerate(checkpoints):
        # Determine if file exists
        if not os.path.exists(path):
            continue
            
        print(f"   -> Adding {path}")
        state = torch.load(path, map_location='cpu')
        
        # Handle different saving formats (some have 'model_state_dict', some are direct)
        if 'model_state_dict' in state:
            params = state['model_state_dict']
        else:
            params = state
            
        for key in params:
            if i == 0:
                avg_state_dict[key] = params[key].clone()
            else:
                avg_state_dict[key] += params[key]
    
    # Compute Mean
    for key in avg_state_dict:
        if avg_state_dict[key].is_floating_point():
            avg_state_dict[key] = avg_state_dict[key] / len(checkpoints)
        else:
            # For integer types like num_batches_tracked, use floor division
            avg_state_dict[key] = avg_state_dict[key] // len(checkpoints)
        
    return avg_state_dict

def main():
    cfg = Config()
    device = cfg.device
    print(f"⚙️ Device: {device}")
    
    # 1. Setup Model & Data
    model = LightweightAgeEstimator(num_classes=cfg.num_classes).to(device)
    _, val_loader, test_loader, _ = get_dataloaders(cfg)
    dldl_tools = DLDLProcessor(cfg)
    
    # 2. Identify Checkpoints (Last 10 epochs)
    # Assuming checkpoints are named checkpoint_epoch_X.pth
    # We look for the files present in the directory
    checkpoints = sorted(glob.glob("checkpoint_epoch_*.pth"))
    
    if not checkpoints:
        print("⚠️ No epoch checkpoints found! Using only best_model if available.")
        checkpoints = []
    else:
        # Filter to keep only the latest ones if there are too many, e.g., last 5-10
        # Since user deleted some, we just take what's there
        print(f"Found {len(checkpoints)} checkpoints candidates.")

    # 3. Strategy A: Soup (Epochs Only)
    print("\n🍵 Strategy A: SWA (Epochs 111-120)")
    soup_a_state = average_weights(checkpoints.copy(), include_best=False)
    mae_a_val = 999.0
    mae_a_test = 999.0
    if soup_a_state:
        model.load_state_dict(soup_a_state)
        mae_a_val = validate(model, val_loader, device, dldl_tools)
        mae_a_test = validate(model, test_loader, device, dldl_tools)
        print(f"✅ Soup A -> Val: {mae_a_val:.4f} | Test: {mae_a_test:.4f}")

    # 4. Strategy B: Soup + Best (Epochs + Best)
    print("\n🥘 Strategy B: SWA + Best Model")
    soup_b_state = average_weights(checkpoints.copy(), include_best=True)
    mae_b_val = 999.0
    mae_b_test = 999.0
    if soup_b_state:
        model.load_state_dict(soup_b_state)
        mae_b_val = validate(model, val_loader, device, dldl_tools)
        mae_b_test = validate(model, test_loader, device, dldl_tools)
        print(f"✅ Soup B -> Val: {mae_b_val:.4f} | Test: {mae_b_test:.4f}")
        
    # 5. Baseline: Best Model Only
    print("\n🥇 Baseline: Best Model Only")
    mae_best_val = 999.0
    mae_best_test = 999.0
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        mae_best_val = validate(model, val_loader, device, dldl_tools)
        mae_best_test = validate(model, test_loader, device, dldl_tools)
        print(f"✅ Best Model -> Val: {mae_best_val:.4f} | Test: {mae_best_test:.4f}")
    
    # 6. Conclusion
    print("\n" + "="*55)
    print("🏆 FINAL SHOWDOWN: VALIDATION vs TEST")
    print("="*55)
    print(f"{'Strategy':<20} | {'Val MAE':<10} | {'Test MAE':<10}")
    print("-" * 55)
    print(f"{'Best Model Only':<20} | {mae_best_val:.4f}     | {mae_best_test:.4f}")
    print(f"{'Soup (Epochs)':<20} | {mae_a_val:.4f}     | {mae_a_test:.4f}")
    print(f"{'Soup (+Best)':<20} | {mae_b_val:.4f}     | {mae_b_test:.4f}")
    print("-" * 55)
    
    # Decision based on Val (Methodologically correct), but we report Test
    scores = {"Best Only": mae_best_val, "Soup (Epochs)": mae_a_val, "Soup (+Best)": mae_b_val}
    winner = min(scores, key=scores.get)
    
    print(f"🎉 Winner (based on Val): {winner}")
    print(f"📝 Report Test MAE: {mae_best_test if 'Best' in winner else (mae_a_test if 'Epochs' in winner else mae_b_test):.4f}")

    # Save if soup won
    if winner == "Soup (Epochs)":
        torch.save(soup_a_state, "final_soup_model.pth")
        print("💾 Saved best soup to: final_soup_model.pth")
    elif winner == "Soup (+Best)":
        torch.save(soup_b_state, "final_soup_model.pth")
        print("💾 Saved best soup to: final_soup_model.pth")
    else:
        print("💾 Best model remains 'best_model.pth'. Kept original.")

if __name__ == "__main__":
    main()
