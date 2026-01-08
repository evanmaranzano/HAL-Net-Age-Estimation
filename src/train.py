import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import Config, ROOT_DIR
from dataset import get_dataloaders
from model import LightweightAgeEstimator
from utils import DLDLProcessor, EMAModel, CombinedLoss, seed_everything
import csv
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# ==========================================
# Reproducibility
# ==========================================
# seed_everything(42)  # Moved to train function


# ==========================================
# MixUp 数据增强函数
# ==========================================
def mixup_data(x, y_dist, y_age, alpha=0.4):
    """
    MixUp 数据增强: 混合两个样本的图像和标签
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y_dist = lam * y_dist + (1 - lam) * y_dist[index]
    # 对真实年龄也做 mixup，用于 Aux Loss
    mixed_y_age = lam * y_age + (1 - lam) * y_age[index]
    
    return mixed_x, mixed_y_dist, mixed_y_age

# ==========================================
# CSV Logger
# ==========================================
class CSVLogger:
    def __init__(self, filepath, headers, resume=False):
        self.filepath = filepath
        self.headers = headers
        if not resume or not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
    def log(self, row_data):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

# ==========================================
# Checkpoint 保存
# ==========================================
def save_checkpoint(state, filename="last_checkpoint.pth"):
    torch.save(state, filename)

# ==========================================
# 主训练函数
# ==========================================
def train(args):
    # Set seed first
    seed = args.seed
    seed_everything(seed)
    
    cfg = Config()
    
    # 🌟 CLI Overrides (Selection Space)
    if args.epochs is not None:
        cfg.epochs = args.epochs
        print(f"🔧 CLI Override: Epochs -> {cfg.epochs}")
        
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
        print(f"🔧 CLI Override: Batch Size -> {cfg.batch_size}")
        
    if args.split is not None:
        cfg.split_protocol = args.split
        print(f"🔧 CLI Override: Split Protocol -> {cfg.split_protocol}")
        
    if args.freeze is not None:
        cfg.freeze_backbone_epochs = args.freeze
        print(f"🔧 CLI Override: Freeze Epochs -> {cfg.freeze_backbone_epochs}")

    # 🌱 Easter Egg: Print Seed Meaning
    if seed in cfg.ACADEMIC_SEEDS:
        print(f"✨ Seed {seed}: {cfg.ACADEMIC_SEEDS[seed]}")

    dldl_tools = DLDLProcessor(cfg)
    
    # ==========================================
    # 2. 准备数据 (Stratified SOTA)
    # ==========================================
    train_loader, val_loader, test_loader, class_weights = get_dataloaders(cfg)
    
    # 打印分布信息
    print(f"Dataset Size: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # 1. 定义模型
    # model = LightweightAgeEstimator(num_classes=cfg.num_classes, dropout=cfg.dropout)
    # Updated for Ablation Support: Pass entire config
    model = LightweightAgeEstimator(cfg)
    model.to(cfg.device)
    
    # 3. 初始化 EMA
    ema = None
    if getattr(cfg, 'use_ema', False):
        print(f"🔄 初始化 EMA (decay={cfg.ema_decay})")
        ema = EMAModel(model, decay=cfg.ema_decay)
        
    # 4. 损失函数 (Combined)
    criterion = CombinedLoss(cfg, weights=class_weights).to(cfg.device)
    
    # 5. 优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay 
    )
    
    # ⚡ AMP Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # 6. 调度器
    # Accelerated Decay: Reach min_lr at Epoch 100, then stay low for 20 epochs (Stable Phase)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=cfg.learning_rate * 0.01)
    
    # --- 断点续训逻辑 ---
    start_epoch = 0
    best_mae = float('inf')
    # Dynamic naming with Seed
    checkpoint_path = os.path.join(ROOT_DIR, f"last_checkpoint_seed{seed}.pth")
    # Dynamic naming: best_model_FADE-Net_HA_DLDL_MSFF_SPP_seed{seed}.pth
    best_model_path = os.path.join(ROOT_DIR, f"best_model_{cfg.project_name}_seed{seed}.pth")
    print(f"🎯 Target Checkpoint Name: {best_model_path}")
    resume_training = False

    if os.path.exists(checkpoint_path):
        print(f"🔄 发现存档 '{checkpoint_path}'，正在恢复...")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
        best_mae = checkpoint.get('best_mae', float('inf'))
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复 EMA
        if ema and 'ema_state_dict' in checkpoint:
            ema.shadow = checkpoint['ema_state_dict']
            print("✅ EMA 状态已恢复")
            
        print(f"✅ 恢复成功！从 Epoch {start_epoch+1} 开始。最佳 MAE: {best_mae:.2f}")
        resume_training = True
    else:
        print("🚀 开始全新训练...")

    # 初始化 Logger (Specific to seed)
    epoch_logger = CSVLogger(os.path.join(ROOT_DIR, f'training_log_seed{seed}.csv'), 
                             ['Epoch', 'Train_Loss', 'Train_MAE', 'Val_Loss', 'Val_MAE', 'LR', 'Time', 'Is_Best'], 
                             resume=resume_training)
    batch_logger = CSVLogger(os.path.join(ROOT_DIR, f'batch_log_seed{seed}.csv'), ['Epoch', 'Batch', 'Total_Loss', 'KL_Loss', 'L1_Loss', 'Rank_Loss'], resume=resume_training)

    # 初始化 TensorBoard Writer
    log_dir = os.path.join(ROOT_DIR, "runs", f"{cfg.project_name}_seed{seed}_{int(time.time())}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📈 TensorBoard 日志目录: {log_dir}")

    print(f"设备: {cfg.device}")
    
    start_time = time.time()
    
    # 🌟 [Innovation] Freeze Backbone Strategy
    # Only train CA modules and Head for the first few epochs
    freeze_epochs = getattr(cfg, 'freeze_backbone_epochs', 0)
    if freeze_epochs > 0:
        if start_epoch < freeze_epochs:
            print(f"❄️  Freeze Strategy Enabled: Backbone will be frozen for first {freeze_epochs} epochs.")
            # Freeze all
            for name, param in model.named_parameters():
                param.requires_grad = False
                
            # Unfreeze classifier
            for param in model.backbone.classifier.parameters():
                param.requires_grad = True
                
            # Unfreeze CoordAtt layers (identified by class name or parameter name)
            # Our CoordAtt modules are inside backbone.features...
            # We can check specific naming or type.
            # Since we modified model.py to inject CoordAtt (class names 'CoordAtt'), we can check modules.
            count_unfrozen = 0
            for name, module in model.named_modules():
                if "CoordAtt" in str(type(module)):
                    for param in module.parameters():
                        param.requires_grad = True
                        count_unfrozen += 1
            
            print(f"    -> Unfrozen Wrapper: Classifier + {count_unfrozen} CoordAtt modules enabled.")
        else:
            print(f"❄️  Freeze Strategy Skipped: Resume Epoch {start_epoch+1} >= Freeze Limit {freeze_epochs}. Backbone remains unfrozen.")

    # 🛡️ Double Check for Safety
    first_param = next(model.backbone.parameters())
    print(f"🔍 检查 Backbone 状态: {'可训练' if first_param.requires_grad else '已冻结'}")

    for epoch in range(start_epoch, cfg.epochs):
        # 🌟 Unfreeze check
        if freeze_epochs > 0 and epoch == freeze_epochs:
            print(f"🔥 Unfreezing Backbone at Epoch {epoch+1} (Fine-tuning begins)...")
            for param in model.parameters():
                param.requires_grad = True
            
            # Optional: Lower LR slightly? Or let cosine scheduler handle it.
            # Cosine is already decaying, so it's fine.

        # 🌟 [Online Hard Distillation] Disable Regularization at later stages
        if epoch >= 105:
            if cfg.use_mixup:
                print(f"🔥 [Epoch {epoch+1}] Hard Distillation Mode: Disabling Mixup!")
                cfg.use_mixup = False
                
            # Disable Random Erasing dynamically by modifying the transform instance in place
            if hasattr(train_loader.dataset, 'transform') and hasattr(train_loader.dataset.transform, 'transforms'):
                for t in train_loader.dataset.transform.transforms:
                    if 'SafeRandomErasing' in str(type(t)) and t.p > 0:
                        print(f"🔥 [Epoch {epoch+1}] Hard Distillation Mode: Disabling Random Erasing!")
                        t.p = 0.0

        # --- 1. 训练 ---
        model.train()
        train_loss = 0.0
        train_mae_sum = 0.0
        train_samples = 0
        
        print(f"\nEpoch [{epoch+1}/{cfg.epochs}] Training (LR: {optimizer.param_groups[0]['lr']:.1e})...")
        
        for batch_idx, (images, target_dists, true_ages) in enumerate(train_loader):
            images = images.to(cfg.device)
            target_dists = target_dists.to(cfg.device)
            true_ages = true_ages.to(cfg.device)
            
            # MixUp
            if cfg.use_mixup and np.random.random() < cfg.mixup_prob:
                images, target_dists, true_ages = mixup_data(
                    images, target_dists, true_ages, alpha=cfg.mixup_alpha
                )
            
            optimizer.zero_grad()
            
            # ⚡ AMP Forward
            with torch.cuda.amp.autocast():
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=1)
                
                # 计算 Combined Loss
                loss, loss_kl, loss_l1, loss_rank = criterion(log_probs, target_dists, true_ages, logits)
            
            # ⚡ AMP Backward
            scaler.scale(loss).backward()
            
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新 EMA
            if ema:
                ema.update()
                
            train_loss += loss.item()
            
            # 计算 MAE (Monitor)
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                pred_ages = dldl_tools.expectation_regression(probs)
                train_mae_sum += torch.sum(torch.abs(pred_ages - true_ages)).item()
                train_samples += true_ages.size(0)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} (KL={loss_kl:.3f}, L1={loss_l1:.3f}, Rank={loss_rank:.3f})")
            
            if batch_idx % 10 == 0:
                batch_logger.log([epoch + 1, batch_idx, loss.item(), loss_kl, loss_l1, loss_rank])
                
                # 📈 TensorBoard Logging (Step)
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/Loss_Total', loss.item(), global_step)
                writer.add_scalar('Train/Loss_KL', loss_kl, global_step)
                writer.add_scalar('Train/Loss_L1', loss_l1, global_step)
                writer.add_scalar('Train/Loss_Rank', loss_rank, global_step)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae_sum / train_samples
        
        # --- 2. 验证 (Validation) ---
        # 如果使用了 EMA，验证时应该使用 EMA 的权重
        if ema:
            ema.apply_shadow()
            print("🛡️切换到 EMA 权重进行验证...")
            
        model.eval()
        mae_sum = 0.0
        val_loss_sum = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for images, target_dists, true_ages in val_loader:
                images = images.to(cfg.device)
                target_dists = target_dists.to(cfg.device)
                true_ages = true_ages.to(cfg.device)
                
                # TTA 验证 (Horizontal Flip)
                # 1. 正常
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                
                # 2. 翻转
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)
                probs_flip = F.softmax(logits_flip, dim=1)
                
                # 3. 融合
                probs = (probs + probs_flip) / 2.0
                
                # 计算 Loss (仅参考，这里只算主KL)
                log_probs = torch.log(probs + 1e-8) 
                val_loss = F.kl_div(log_probs, target_dists, reduction='batchmean')
                val_loss_sum += val_loss.item()
                
                pred_ages = dldl_tools.expectation_regression(probs)
                mae_sum += torch.sum(torch.abs(pred_ages - true_ages)).item()
                total_samples += true_ages.size(0)
        
        # 验证结束，如果用了 EMA，恢复原始权重以便继续训练
        if ema:
            ema.restore()
            print("🛡️恢复原始权重继续训练...")
            
        val_mae = mae_sum / total_samples
        avg_val_loss = val_loss_sum / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{cfg.epochs}] | "
              f"T_Loss: {avg_train_loss:.4f} | T_MAE: {avg_train_mae:.2f} | "
              f"V_MAE: {val_mae:.2f}")

        # --- 3. 保存最佳模型 ---
        is_best = False
        if val_mae < best_mae:
            print(f"🏆 新纪录！MAE {best_mae:.2f} -> {val_mae:.2f}")
            best_mae = val_mae
            is_best = True
            
            # 如果用了 EMA，保存 EMA 后的权重为 best_model.pth
            if ema:
                ema.apply_shadow()
                torch.save(model.state_dict(), best_model_path)
                ema.restore()
            else:
                torch.save(model.state_dict(), best_model_path)
                
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        epoch_logger.log([epoch + 1, avg_train_loss, avg_train_mae, avg_val_loss, val_mae, current_lr, elapsed, int(is_best)])
        
        # 📈 TensorBoard Logging (Epoch)
        writer.add_scalar('Epoch/Train_Loss', avg_train_loss, epoch + 1)
        writer.add_scalar('Epoch/Train_MAE', avg_train_mae, epoch + 1)
        writer.add_scalar('Epoch/Val_Loss', avg_val_loss, epoch + 1)
        writer.add_scalar('Epoch/Val_MAE', val_mae, epoch + 1)
        
        if epoch < 100:
            scheduler.step()
        else:
             # Maintain eta_min for Stable Phase (101-120)
             pass
        
        # 保存断点
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'best_mae': best_mae
        }
        if ema:
            checkpoint_dict['ema_state_dict'] = ema.shadow
            
        save_checkpoint(checkpoint_dict, filename=checkpoint_path)
        
        # --- Manual SWA Strategy ---
        # Save checkpoints for the last 10 epochs
        if epoch >= cfg.epochs - 10:
            swa_filename = os.path.join(ROOT_DIR, f"checkpoint_seed{seed}_epoch_{epoch+1}.pth")
            print(f"💾 Saving SWA Checkpoint: {swa_filename}")
            save_checkpoint(checkpoint_dict, filename=swa_filename)
    

    # ==========================================
    # 🏁 Final Test Set Evaluation
    # ==========================================
    print("\n" + "="*50)
    print("🏁 Final Evaluation on TEST SET")
    print("="*50)
    
    # Load Best Model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=cfg.device))
        print(f"📂 Loaded Best Model from {best_model_path}")
    
    model.eval()
    test_mae = 0.0
    count = 0
    rank_arange = torch.arange(cfg.num_classes).to(cfg.device) # Define rank_arange
    
    with torch.no_grad():
        for images, labels, ages in test_loader:
            images, labels, ages = images.to(cfg.device), labels.to(cfg.device), ages.to(cfg.device)
            # TTA Evaluation
            # 1. Normal
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            
            # 2. Horizontal Flip
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip)
            probs_flip = F.softmax(logits_flip, dim=1)
            
            # 3. Fuse
            probs = (probs + probs_flip) / 2.0
            
            # Predict
            output_ages = torch.sum(probs * rank_arange.float(), dim=1)
            
            # MAE
            mae = torch.abs(output_ages - ages).sum().item()
            test_mae += mae
            count += images.size(0)
            
    final_test_mae = test_mae / count
    print(f"🏆 Final Test MAE: {final_test_mae:.4f}")
    
    # Save Final Result
    with open(os.path.join(ROOT_DIR, f"final_result_seed{seed}.txt"), "w") as f:
        f.write(f"Test MAE: {final_test_mae:.4f}\n")
        
    writer.close()

if __name__ == "__main__":
    import sys
    import subprocess
    import re

    # Helper function for Batch Mode (Run All)
    def run_training_subprocess(seed):
        print(f"\n🚀 Starting subprocess for seed {seed}...")
        # Use sys.executable to ensure we use the same python interpreter
        # Use sys.argv[0] to refer to this script (train.py)
        cmd = [sys.executable, sys.argv[0], "--seed", str(seed)]
        
        # Pass through other common args if needed, or enforce defaults for benchmarks
        # For 'Run All', we usually want standard settings, so we just pass seed.
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        
        mae = None
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                if "Final Test MAE:" in output:
                    try:
                        mae = float(output.strip().split(":")[-1].strip())
                    except:
                        pass
        
        rc = process.poll()
        if rc != 0:
            print(f"❌ Training failed for seed {seed}")
            return None
            
        # Fallback check file
        if mae is None:
            # Assuming ROOT_DIR is defined and available
            result_file = os.path.join(ROOT_DIR, f"final_result_seed{seed}.txt")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    content = f.read()
                    match = re.search(r"Test MAE:\s*([\d\.]+)", content)
                    if match:
                        mae = float(match.group(1))
        return mae

    # --- CLI Handling ---
    # Case 1: Arguments provided -> Run Training Immediately
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="FADE-Net Training Launcher")
        parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
        parser.add_argument('--epochs', type=int, help='Override total training epochs')
        parser.add_argument('--batch_size', type=int, help='Override batch size')
        parser.add_argument('--split', type=str, choices=['90-5-5', '72-8-20'], help="Select Split Protocol")
        parser.add_argument('--freeze', type=int, dest='freeze', help='Override backbone freeze epochs')
        parser.add_argument('--freeze_backbone_epochs', type=int, dest='freeze_alias', help='Alias for --freeze') 
        
        args = parser.parse_args()
        
        # Handle alias
        if args.freeze_alias is not None:
            args.freeze = args.freeze_alias

        torch.backends.cudnn.benchmark = True
        train(args)
        sys.exit(0)

    # Case 2: No Arguments -> Interactive Menu
    print("="*60)
    print("🎮 FADE-Net Interactive Training Launcher")
    print("="*60)
    print("1. [Default]  Run Standard Benchmark (Seed 42, 90-5-5)")
    print("2. [SOTA]     Run 2026 Academic Seed (Seed 2026, 90-5-5)")
    print("3. [Batch]    Run All Academic Seeds (42, 3407, 2026, 1337, 1106)")
    print("4. [Custom]   Configure Manually")
    print("q. [Quit]     Exit")
    print("-" * 60)
    
    try:
        choice = input("👉 Select mode [1-4/q]: ").strip().lower()
        
        if choice == '1' or choice == '':
            print("\n🚀 Selected: Standard Benchmark (Seed 42)")
            # Simulate args
            class Args:
                seed = 42
                epochs = None
                batch_size = None
                split = None
                freeze = None
            train(Args())
            
        elif choice == '2':
            print("\n🚀 Selected: SOTA 2026 (Seed 2026)")
            class Args:
                seed = 2026
                epochs = None
                batch_size = None
                split = None
                freeze = None
            train(Args())

        elif choice == '3':
            print("\n🚀 Selected: Run All Academic Seeds")
            seeds = [42, 3407, 2026, 1337, 1106]
            results = {}
            for s in seeds:
                mae = run_training_subprocess(s)
                if mae is not None:
                    results[s] = mae
            
            print("\n" + "=" * 60)
            print("📊 Final Batch Report")
            print("=" * 60)
            if results:
                maes = list(results.values())
                mean_mae = np.mean(maes)
                std_mae = np.std(maes)
                print(f"{'Seed':<10} | {'Test MAE':<10}")
                print("-" * 25)
                for s, m in results.items():
                    print(f"{s:<10} | {m:.4f}")
                print("-" * 25)
                print(f"\n🏆 Average Test MAE: {mean_mae:.4f} ± {std_mae:.4f}")
            else:
                print("No successful runs.")

        elif choice == '4':
            print("\n🔧 Custom Configuration Mode:")
            s = input("   - Seed [42]: ").strip() or '42'
            sp_choice = input("   - Split (1: 90-5-5, 2: 72-8-20) [1]: ").strip()
            split = '72-8-20' if sp_choice == '2' else '90-5-5'
            ep = input("   - Epochs [Default]: ").strip()
            fz = input("   - Freeze Epochs [Default]: ").strip()
            
            class Args:
                seed = int(s)
                split = split
                epochs = int(ep) if ep else None
                batch_size = None
                freeze = int(fz) if fz else None
            
            train(Args())
            
        elif choice == 'q':
            pass
            
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
        sys.exit(0)


