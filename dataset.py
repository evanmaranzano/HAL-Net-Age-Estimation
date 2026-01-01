import os
import torch
import numpy as np
import json
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
from utils import DLDLProcessor, FaceAligner
from config import Config
from collections import Counter, defaultdict
from scipy.ndimage import gaussian_filter1d

# ==========================================
# Collate Function
# ==========================================
def my_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# ==========================================
# 0. Stratified Split Strategy (The "Platinum" Choice)
# ==========================================
def get_stratified_split(dataset, all_ages, split_ratios=(0.90, 0.05, 0.05), save_path="dataset_split_stratified.json"):
    """
    Perform Stratified Sampling based on age labels.
    Ensures 90/5/5 split holds true for *every single age class*.
    """
    assert abs(sum(split_ratios) - 1.0) < 1e-5, "Split ratios must sum to 1"
    
    # Check for existing split
    if os.path.exists(save_path):
        print(f"📄 Loading existing stratified split from {save_path}...")
        try:
            with open(save_path, "r") as f:
                indices_dict = json.load(f)
            train_idx = indices_dict['train']
            val_idx = indices_dict['val']
            test_idx = indices_dict['test']
            print(f"✅ Loaded: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
            return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
        except Exception as e:
            print(f"⚠️ Load failed ({e}), regenerating...")

    print("⚖️ Performing Stratified Sampling (90/5/5 per age)...")
    
    # Group indices by age
    indices_by_age = defaultdict(list)
    for idx, age in enumerate(all_ages):
        age_int = int(round(age))
        indices_by_age[age_int].append(idx)
        
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Fixed seed for generation
    random.seed(42)
    
    for age, indices in indices_by_age.items():
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])
        # Remaining goes to test (handles rounding errors)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
        
    print(f"✅ Stratified Split Done.")
    print(f"   Train: {len(train_indices)}")
    print(f"   Val:   {len(val_indices)}")
    print(f"   Test:  {len(test_indices)}")
    
    # Save to JSON
    save_data = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f)
    print(f"💾 Split saved to {save_path}")
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

# ==========================================
# 2. AFAD Dataset
# ==========================================
class AFADDataset(Dataset):
    def __init__(self, root_dir, transform=None, config=None):
        self.transform = transform
        self.config = config
        self.dldl_proc = DLDLProcessor(config)
        self.image_paths = []
        self.ages = []
        
        if not os.path.exists(root_dir):
            print(f"⚠️ [AFAD] Path not found: {root_dir}")
        else:
            print("⏳ Scanning AFAD...")
            # Sorted ensures determinstic order before shuffling
            for age_folder in sorted(os.listdir(root_dir)):
                age_path = os.path.join(root_dir, age_folder)
                if os.path.isdir(age_path) and age_folder.isdigit():
                    age = int(age_folder)
                    # Use Strict range
                    if age < config.min_age or age > config.max_age:
                        continue
                    
                    for gender_folder in sorted(os.listdir(age_path)):
                        gender_path = os.path.join(age_path, gender_folder)
                        if os.path.isdir(gender_path):
                            for img_name in sorted(os.listdir(gender_path)):
                                if img_name.lower().endswith(('.jpg', '.png')):
                                    self.image_paths.append(os.path.join(gender_path, img_name))
                                    self.ages.append(float(age))
            print(f"✅ AFAD Loaded: {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        age = self.ages[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            return None
        if self.transform:
            image = self.transform(image)
        label_dist = self.dldl_proc.generate_label_distribution(age)
        return image, label_dist, torch.tensor(age, dtype=torch.float32)

# ==========================================
# 3. AAF Dataset
# ==========================================
class AAFDataset(Dataset):
    def __init__(self, root_dir, transform=None, config=None):
        self.transform = transform
        self.config = config
        self.dldl_proc = DLDLProcessor(config)
        self.image_paths = []
        self.ages = []
        
        if not os.path.exists(root_dir):
            print(f"⚠️ [AAF] Path not found: {root_dir}")
        else:
            print("⏳ Scanning AAF...")
            for img_name in sorted(os.listdir(root_dir)):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    try:
                        if 'A' in img_name.upper():
                            age_str = img_name.upper().split('A')[1].split('.')[0]
                            age = int(age_str)
                            if age < config.min_age or age > config.max_age:
                                continue
                            self.image_paths.append(os.path.join(root_dir, img_name))
                            self.ages.append(float(age))
                    except:
                        continue
            print(f"✅ AAF Loaded: {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        age = self.ages[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            return None
        if self.transform:
            image = self.transform(image)
        label_dist = self.dldl_proc.generate_label_distribution(age)
        return image, label_dist, torch.tensor(age, dtype=torch.float32)

# ==========================================
# Subset with Transform
# ==========================================
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        item = self.subset[idx]
        if item is None: return None
        image, label_dist, age = item
        if self.transform:
            image = self.transform(image)
        return image, label_dist, age
        
    def __len__(self):
        return len(self.subset)

# ==========================================
# LDS Weights
# ==========================================
def calculate_lds_weights(ages, config, smoothing_sigma=5):
    print("⚖️ Calculating LDS Weights...")
    age_counts = Counter(ages)
    hist = np.zeros(config.num_classes)
    for age, count in age_counts.items():
        idx = int(round(age))
        if 0 <= idx < config.num_classes:
            hist[idx] = count
    
    smooth_hist = gaussian_filter1d(hist, sigma=smoothing_sigma)
    weights = 1.0 / (smooth_hist + 1e-5)
    weights = weights / np.mean(weights)
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(config.device)
    print("✅ LDS Weights Ready.")
    return weights_tensor

# ==========================================
# Main: Get DataLoaders
# ==========================================
def get_dataloaders(config):
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("=" * 60)
    print("🚀 Loading Dataset (AFAD + AAF Only)")
    print("=" * 60)
    
    all_datasets = []
    all_ages = []
    
    # 1. AFAD
    if hasattr(config, 'afad_dir') and os.path.exists(config.afad_dir):
        afad = AFADDataset(config.afad_dir, config=config)
        if len(afad) > 0:
            all_datasets.append(afad)
            all_ages.extend(afad.ages)
            
    # 2. AAF
    if hasattr(config, 'aaf_dir') and os.path.exists(config.aaf_dir):
        aaf = AAFDataset(config.aaf_dir, config=config)
        if len(aaf) > 0:
            all_datasets.append(aaf)
            all_ages.extend(aaf.ages)
            
    if not all_datasets:
        raise ValueError("No datasets found! Check config paths.")
        
    # LDS Weights based on FULL distribution
    class_weights = None
    if getattr(config, 'use_reweighting', False):
        class_weights = calculate_lds_weights(all_ages, config)
        
    full_dataset = ConcatDataset(all_datasets)
    print(f"\n📦 Total Images: {len(full_dataset)}")
    
    # Stratified Split
    train_subset, val_subset, test_subset = get_stratified_split(
        full_dataset, 
        all_ages, 
        split_ratios=(0.90, 0.05, 0.05)
    )
    
    # Apply Transforms
    train_set = SubsetWithTransform(train_subset, transform=train_transform)
    val_set = SubsetWithTransform(val_subset, transform=val_transform)
    test_set = SubsetWithTransform(test_subset, transform=val_transform)
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, 
                              num_workers=config.num_workers, pin_memory=True, collate_fn=my_collate_fn)
                              
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, 
                            num_workers=config.num_workers, pin_memory=True, collate_fn=my_collate_fn)
                            
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, 
                             num_workers=config.num_workers, pin_memory=True, collate_fn=my_collate_fn)
                             
    return train_loader, val_loader, test_loader, class_weights