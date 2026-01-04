import subprocess
import os
import re
import numpy as np
import sys
from src.config import Config

# Define seeds from Shared Config
SEEDS = list(Config.ACADEMIC_SEEDS.keys())
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable

def run_training(seed):
    print(f"\n🌱 Starting training with seed {seed}...")
    cmd = [PYTHON_EXE, os.path.join(PROJECT_ROOT, "src", "train.py"), "--seed", str(seed)]
    
    # Run and stream output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    
    mae = None
    
    # Read output line by line
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            # Capture Final Test MAE
            if "Final Test MAE:" in output:
                try:
                    mae = float(output.strip().split(":")[-1].strip())
                except:
                    pass
    
    rc = process.poll()
    if rc != 0:
        print(f"❌ Training failed for seed {seed}")
        return None
        
    # Double check file if not found in stdout
    if mae is None:
        result_file = os.path.join(PROJECT_ROOT, f"final_result_seed{seed}.txt")
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                content = f.read()
                # format: Test MAE: 3.xxxx
                match = re.search(r"Test MAE:\s*([\d\.]+)", content)
                if match:
                    mae = float(match.group(1))
    
    return mae

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run training with academic seeds.")
    parser.add_argument('--seed', type=int, help='Run a specific seed (e.g., 42). If not set, runs all default seeds by default (unless --custom is used).')
    parser.add_argument('--custom', type=int, help='Run a custom seed.')
    parser.add_argument('--all', action='store_true', help='Force run all default seeds: [42, 2024, 3407]')
    args = parser.parse_args()

    print("=" * 60)
    print("🔬 Academic Rigour Protocol: Seed Experiment Runner")
    print("=" * 60)
    
    seeds_to_run = []
    
    # 1. Check for Command Line Arguments
    if args.custom is not None:
        seeds_to_run = [args.custom]
        print(f"👉 Mode: Custom Seed ({args.custom})")
    elif args.seed is not None:
        seeds_to_run = [args.seed]
        print(f"👉 Mode: Single Seed ({args.seed})")
    elif args.all:
         seeds_to_run = SEEDS
         print(f"👉 Mode: All Default Seeds {SEEDS}")
    
    # 2. If No Arguments, Enter Interactive Mode
    else:
        print("\n👋 No arguments detected. Please select a mode:")
        print("   [1] Run Default Seed (42)  -> Quick Verification")
        print("   [2] Run Custom Seed        -> You specify the number")
        print(f"   [3] Run All Academic Seeds -> {SEEDS} (For Paper)")
        print("   [q] Quit")
        
        try:
            choice = input("\n👉 Enter choice [1/2/3/q]: ").strip().lower()
            
            if choice == '1':
                seeds_to_run = [42]
                print("👉 Selected: Seed 42")
            elif choice == '2':
                s = input("👉 Enter integer seed: ").strip()
                if s.isdigit():
                    seeds_to_run = [int(s)]
                else:
                    print("❌ Invalid integer. Exiting.")
                    sys.exit(1)
            elif choice == '3':
                seeds_to_run = SEEDS
                print(f"👉 Selected: All Seeds {SEEDS}")
            elif choice == 'q':
                print("👋 Exiting.")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Defaulting to Seed 42.")
                seeds_to_run = [42]
                
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            sys.exit(0)

    results = {}
    
    for seed in seeds_to_run:
        mae = run_training(seed)
        if mae is not None:
            results[seed] = mae
            print(f"✅ Seed {seed} Finished. MAE: {mae:.4f}")
        else:
            print(f"❌ Seed {seed} Failed.")
    
    # Only show summary if we ran more than 1 seed
    if len(seeds_to_run) > 1:
        print("\n" + "=" * 60)
        print("📊 Final Report")
        print("=" * 60)
        
        maes = list(results.values())
        if not maes:
            print("No successful runs.")
            return

        mean_mae = np.mean(maes)
        std_mae = np.std(maes)
        
        print(f"{'Seed':<10} | {'Test MAE':<10}")
        print("-" * 25)
        for seed, mae in results.items():
            print(f"{seed:<10} | {mae:.4f}")
        print("-" * 25)
        
        print(f"\n🏆 Average Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
        print("=" * 60)
        print("\n📝 LaTeX Snippet:")
        print(f"Our method achieves an MAE of ${mean_mae:.3f} \\pm {std_mae:.3f}$ over {len(seeds_to_run)} runs.")

if __name__ == "__main__":
    main()
