import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import argparse # Added
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

def average_weights(checkpoints, include_best=False, best_path=None):
    if include_best and best_path and os.path.exists(best_path):
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

def main(seed):
    print(f"\n🥣 Starting Model Soup for Seed {seed}...")
    cfg = Config()
    device = cfg.device
    print(f"⚙️ Device: {device}")
    
    # 1. Setup Model & Data
    model = LightweightAgeEstimator(config=cfg).to(device)
    _, val_loader, test_loader, _ = get_dataloaders(cfg)
    dldl_tools = DLDLProcessor(cfg)
    
    # 2. Identify Checkpoints (Last 10 epochs)
    # Search for specific seed patterns
    pattern = f"checkpoint_seed{seed}_epoch_*.pth"
    checkpoints = sorted(glob.glob(pattern))
    
    if not checkpoints:
        print(f"⚠️ No epoch checkpoints found for pattern '{pattern}'!")
        print("Using only best_model if available.")
        checkpoints = []
    else:
        print(f"Found {len(checkpoints)} checkpoints candidates for Seed {seed}.")

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
    # Dynamic naming: best_model_FADE-Net_HA_DLDL_MSFF_SPP_seed{seed}.pth
    best_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"best_model_{cfg.project_name}_seed{seed}.pth")
    soup_b_state = average_weights(checkpoints.copy(), include_best=True, best_path=best_model_path)
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
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        mae_best_val = validate(model, val_loader, device, dldl_tools)
        mae_best_test = validate(model, test_loader, device, dldl_tools)
        print(f"✅ Best Model -> Val: {mae_best_val:.4f} | Test: {mae_best_test:.4f}")
    else:
        print(f"❌ Best Model not found at: {best_model_path}")
    
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
    output_filename = f"final_soup_model_seed{seed}.pth"
    if winner == "Soup (Epochs)":
        torch.save(soup_a_state, output_filename)
        print(f"💾 Saved best soup to: {output_filename}")
    elif winner == "Soup (+Best)":
        torch.save(soup_b_state, output_filename)
        print(f"💾 Saved best soup to: {output_filename}")
    else:
        print(f"💾 Best model remains '{os.path.basename(best_model_path)}'. Kept original.")

if __name__ == "__main__":
    import sys
    
    # Check for CLI args first
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Seed used for training')
    args = parser.parse_args()

    if args.seed is not None:
        main(args.seed)
    else:
        # Interactive Mode
        print("="*60)
        print("🍵 FADE-Net Model Soup Kitchen")
        print("="*60)
        
        # Scan for available seeds based on checkpoints
        # Look for checkpoint_seed{SEED}_epoch_*.pth
        all_checkpoints = glob.glob("checkpoint_seed*_epoch_*.pth")
        seeds = set()
        for cp in all_checkpoints:
            try:
                # Naive parse: checkpoint_seed123_epoch_115.pth
                # Split by '_' -> ['checkpoint', 'seed123', 'epoch', '115.pth']
                parts = cp.split('_')
                if len(parts) >= 2 and parts[1].startswith('seed'):
                    s = int(parts[1].replace('seed', ''))
                    seeds.add(s)
            except:
                continue
        
        sorted_seeds = sorted(list(seeds), key=lambda x: (0 if x == 42 else 1, x))
        
        print("🔍 Found soup ingredients for seeds:")
        menu_map = {}
        for idx, s in enumerate(sorted_seeds):
            print(f"   {idx+1}. [Seed {s}]")
            menu_map[str(idx+1)] = s
            
        print(f"   m. [Manual] Enter Seed ID Manually")
        print("   q. [Quit]   Exit")
        print("-" * 60)
        
        try:
            choice = input(f"👉 Select seed to cook [1-{len(sorted_seeds)}]: ").strip().lower()
            
            if choice == 'q':
                sys.exit(0)
            elif choice == 'm':
                 manual_seed = int(input("👉 Enter Seed ID: ").strip())
                 main(manual_seed)
            elif choice in menu_map:
                main(menu_map[choice])
            else:
                 print("❌ Invalid choice.")
                 
        except KeyboardInterrupt:
             sys.exit(0)
        except ValueError:
             print("❌ Invalid input.")

