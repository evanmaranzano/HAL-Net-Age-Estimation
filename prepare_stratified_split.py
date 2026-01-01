import os
import json
import random
import numpy as np
from collections import defaultdict, Counter
from config import Config
from tqdm import tqdm

def scan_afad(root_dir, config):
    image_paths = []
    ages = []
    print(f"📂 [AFAD] Scanning {root_dir}...")
    
    if not os.path.exists(root_dir):
        print(f"❌ Path not found: {root_dir}")
        return [], []

    # Sorted for determinism
    for age_folder in sorted(os.listdir(root_dir)):
        age_path = os.path.join(root_dir, age_folder)
        if os.path.isdir(age_path) and age_folder.isdigit():
            age = int(age_folder)
            
            # Filter strictly by Config range
            if age < config.min_age or age > config.max_age:
                continue
            
            for gender_folder in sorted(os.listdir(age_path)):
                gender_path = os.path.join(age_path, gender_folder)
                if os.path.isdir(gender_path):
                    for img_name in sorted(os.listdir(gender_path)):
                        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                            image_paths.append(os.path.join(gender_path, img_name))
                            ages.append(age)
    
    print(f"✅ [AFAD] Found {len(image_paths)} images (Age {config.min_age}-{config.max_age})")
    return image_paths, ages

def scan_aaf(root_dir, config):
    image_paths = []
    ages = []
    print(f"📂 [AAF] Scanning {root_dir}...")
    
    if not os.path.exists(root_dir):
        print(f"❌ Path not found: {root_dir}")
        return [], []

    for img_name in sorted(os.listdir(root_dir)):
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            try:
                # Format: 12A34.jpg -> Age 12 (Example, actual AAF format varies, usually ID_A<age>.jpg)
                # Assuming standard 'A' delimiter from previous context
                if 'A' in img_name.upper():
                    age_part = img_name.upper().split('A')[1]
                    # Handle "25.jpg" or "25_female..."
                    age_str = ""
                    for char in age_part:
                        if char.isdigit():
                            age_str += char
                        else:
                            break
                    
                    if not age_str: continue
                    age = int(age_str)
                    
                    if age < config.min_age or age > config.max_age:
                        continue
                        
                    image_paths.append(os.path.join(root_dir, img_name))
                    ages.append(age)
            except Exception:
                continue
                
    print(f"✅ [AAF] Found {len(image_paths)} images (Age {config.min_age}-{config.max_age})")
    return image_paths, ages

def main():
    cfg = Config()
    print(f"🔧 Config: Max Age={cfg.max_age}, Min Age={cfg.min_age}")
    print("="*60)
    
    # 1. Collect Data
    afad_imgs, afad_ages = scan_afad(cfg.afad_dir, cfg)
    aaf_imgs, aaf_ages = scan_aaf(cfg.aaf_dir, cfg)
    
    all_imgs = afad_imgs + aaf_imgs
    all_ages = afad_ages + aaf_ages
    
    total_count = len(all_imgs)
    print("="*60)
    print(f"📦 Total Collection: {total_count} images")
    
    if total_count == 0:
        print("❌ No images found! Check paths in config.py")
        return

    # 2. Stratified Split Logic
    print("\n⚖️  Performing Stratified Split (90/5/5)...")
    
    # Group indices by age
    indices_by_age = defaultdict(list)
    for idx, age in enumerate(all_ages):
        indices_by_age[int(age)].append(idx)
        
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Ratios
    r_train, r_val, r_test = 0.90, 0.05, 0.05
    
    random.seed(42) # Strict Seed for splitting
    
    stats_train = defaultdict(int)
    stats_val = defaultdict(int)
    stats_test = defaultdict(int)
    
    # Iterate EVERY age label to ensure coverage
    sorted_ages = sorted(indices_by_age.keys())
    
    for age in sorted_ages:
        idxs = indices_by_age[age]
        random.shuffle(idxs)
        
        n = len(idxs)
        n_train = int(n * r_train)
        n_val = int(n * r_val)
        # Force at least 1 val and 1 test if we have enough samples (e.g. > 20)
        # Or blindly trust ratio. For SOTA, blind ratio is standard, but for rare classes we might get 0 val.
        # Let's stick to standard math to preserve pure distribution.
        
        t_idxs = idxs[:n_train]
        v_idxs = idxs[n_train : n_train + n_val]
        te_idxs = idxs[n_train + n_val:]
        
        train_indices.extend(t_idxs)
        val_indices.extend(v_idxs)
        test_indices.extend(te_idxs)
        
        stats_train[age] = len(t_idxs)
        stats_val[age] = len(v_idxs)
        stats_test[age] = len(te_idxs)

    # 3. Save JSON
    # IMPORTANT: The dataset loading order in train.py is AFAD then AAF.
    # ConcatDataset concatenates them in order.
    # Our `all_imgs` list is [AFAD..., AAF...].
    # So the indices [0...N] perfectly map to ConcatDataset([AFAD, AAF]).
    # This JSON is valid for `dataset.py` logic.
    
    output_data = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
        'meta': {
            'total': total_count,
            'max_age': cfg.max_age,
            'source': 'AFAD+AAF'
        }
    }
    
    save_path = "dataset_split_stratified.json"
    with open(save_path, "w") as f:
        json.dump(output_data, f)
        
    print("\n" + "="*60)
    print(f"💾 Saved to {save_path}")
    print(f"   Train: {len(train_indices)} ({len(train_indices)/total_count:.1%})")
    print(f"   Val:   {len(val_indices)}   ({len(val_indices)/total_count:.1%})")
    print(f"   Test:  {len(test_indices)}   ({len(test_indices)/total_count:.1%})")
    print("="*60)
    
    # 4. Detailed Report (Optional, saving to text)
    with open("split_report_stratified.txt", "w") as f:
        f.write("Age,Train,Val,Test\n")
        f.write(f"Total,{len(train_indices)},{len(val_indices)},{len(test_indices)}\n")
        for age in sorted_ages:
            f.write(f"{age},{stats_train[age]},{stats_val[age]},{stats_test[age]}\n")
            
    print("📝 Saved distribution report to split_report_stratified.txt")
    print("🚀 You can now run 'python train.py' directly!")

if __name__ == "__main__":
    main()
