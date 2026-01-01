import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# ================= 配置区域 =================
SAVE_DIR = 'plots'
plt.style.use('seaborn-v0_8-paper')

# 统一的论文级字体配置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2.5 
})

def load_real_data():
    if not (os.path.exists('training_log.csv') and os.path.exists('batch_log.csv')):
        print("❌ 错误: 未找到 CSV 文件")
        return None, None
    print("📂 读取数据中...")
    df_epoch = pd.read_csv('training_log.csv')
    df_batch = pd.read_csv('batch_log.csv')
    return df_epoch, df_batch

def add_best_model_line(ax_plt, epoch, label_y_pos=None, color='#333333'):
    """辅助函数：添加最佳模型垂直线"""
    ax_plt.axvline(x=epoch, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    
    ymin, ymax = ax_plt.get_ylim()
    text_pos = ymax - (ymax - ymin) * 0.05 if label_y_pos is None else label_y_pos
    
    ax_plt.text(epoch, text_pos, ' Best Checkpoint', rotation=90, 
                verticalalignment='top', fontsize=10, color=color, alpha=0.8)

def plot_thesis_suite():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    df_epoch, df_batch = load_real_data()
    if df_epoch is None: return

    # 计算全局最佳轮次 (基于 Val_MAE)
    best_idx = df_epoch['Val_MAE'].idxmin()
    best_epoch = df_epoch.loc[best_idx, 'Epoch']
    best_mae_val = df_epoch.loc[best_idx, 'Val_MAE']
    
    print(f"🚀 开始生成 8 张独立图表 -> {SAVE_DIR}/")
    print(f"💡 最佳模型出现在第 {best_epoch} 轮 (MAE={best_mae_val:.4f})")

    # ==========================================
    # 图 1: Loss 收敛曲线 (基础版)
    # ==========================================
    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    plt.plot(df_epoch['Epoch'], df_epoch['Train_Loss'], label='Train Loss', color='#2878B5')
    plt.plot(df_epoch['Epoch'], df_epoch['Val_Loss'], label='Val Loss', color='#D76364', linestyle='--')
    add_best_model_line(ax1, best_epoch)
    plt.title('Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(frameon=True, fancybox=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/1_loss_curve.png')
    plt.close()

    # ==========================================
    # 图 2: MAE 性能曲线
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.plot(df_epoch['Epoch'], df_epoch['Train_MAE'], label='Train MAE', color='#9AC9DB')
    plt.plot(df_epoch['Epoch'], df_epoch['Val_MAE'], label='Val MAE', color='#C82423', linestyle='--')
    plt.axvline(x=best_epoch, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    plt.scatter(best_epoch, best_mae_val, color='black', s=60, zorder=5)
    plt.annotate(f'Best MAE: {best_mae_val:.2f}\n(Epoch {best_epoch})', 
                 xy=(best_epoch, best_mae_val), 
                 xytext=(best_epoch + 5, best_mae_val + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    plt.title('Model Performance (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend(frameon=True, fancybox=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/2_mae_curve.png')
    plt.close()

    # ==========================================
    # 图 3: 学习率调度
    # ==========================================
    plt.figure(figsize=(8, 4))
    plt.plot(df_epoch['Epoch'], df_epoch['LR'], color='#6D6D6D', alpha=0.8)
    plt.fill_between(df_epoch['Epoch'], df_epoch['LR'], color='#6D6D6D', alpha=0.1)
    plt.axvline(x=best_epoch, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/3_lr_schedule.png')
    plt.close()

    # ==========================================
    # 图 4: 泛化差距分析
    # ==========================================
    gap = df_epoch['Val_Loss'] - df_epoch['Train_Loss']
    plt.figure(figsize=(8, 5))
    ax4 = plt.gca()
    plt.plot(df_epoch['Epoch'], gap, color='#845EC2', label='Generalization Gap')
    plt.fill_between(df_epoch['Epoch'], gap, 0, color='#845EC2', alpha=0.15)
    z = np.polyfit(df_epoch['Epoch'], gap, 1)
    p = np.poly1d(z)
    plt.plot(df_epoch['Epoch'], p(df_epoch['Epoch']), "k--", alpha=0.5, linewidth=1, label='Gap Trend')
    add_best_model_line(ax4, best_epoch)
    plt.title('Generalization Gap Dynamics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference ($Val - Train$)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/4_generalization_gap.png')
    plt.close()

    # ==========================================
    # 图 5: Batch 稳定性 (趋势图)
    # ==========================================
    plt.figure(figsize=(12, 4))
    global_steps = range(len(df_batch))
    plt.plot(global_steps, df_batch['Total_Loss'], color='#555555', alpha=0.3, linewidth=0.5, label='Raw Batch Loss')
    window = 100
    if len(df_batch) > window:
        trend = df_batch['Total_Loss'].rolling(window).mean()
        plt.plot(global_steps, trend, color='#C82423', linewidth=1.5, label=f'Trend (MA={window})')
    plt.title('Training Stability (Batch Level)')
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    limit = df_batch['Total_Loss'].iloc[int(len(df_batch)*0.01):].quantile(0.999) * 1.1
    plt.ylim(0, limit)
    plt.legend(loc='upper right', frameon=True)
    plt.margins(x=0)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/5_batch_stability.png')
    plt.close()

    # ==========================================
    # [NEW] 图 6: 训练时间效率分析 (Time Efficiency)
    # ==========================================
    # 计算每个 Epoch 的耗时 (处理断点续训的情况)
    time_deltas = []
    prev_time = 0
    print("\n⏱️ 正在分析训练耗时 (检测断点)...")
    for idx, t in enumerate(df_epoch['Time']):
        epoch_num = df_epoch.loc[idx, 'Epoch']
        if t < prev_time: # 发生了重启
            delta = t
            print(f"  -> Epoch {epoch_num}: 检测到时间重置 (Time={t:.1f}s) -> 判定为重启后首轮")
        else:
            delta = t - prev_time
        
        time_deltas.append(delta)
        prev_time = t
    
    avg_time = np.mean(time_deltas)
    print(f"  -> 平均每轮耗时: {avg_time:.2f} 秒")

    plt.figure(figsize=(8, 5))
    plt.plot(df_epoch['Epoch'], time_deltas, marker='o', markersize=4, color='#2E8B57', alpha=0.8)
    plt.axhline(y=avg_time, color='#2E8B57', linestyle='--', alpha=0.5, label=f'Avg: {avg_time:.1f}s')
    plt.title('Training Time Cost per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Duration (seconds)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/6_time_efficiency.png')
    plt.close()

    # ==========================================
    # [NEW] 图 7: Batch Loss 分布 (Boxplot)
    # ==========================================
    # 展示每个 Epoch 的 Loss 分布，观察收敛的方差变化
    plt.figure(figsize=(10, 6))
    # 为了防止 Epoch 太多导致箱线图太挤，我们每隔几个 Epoch 采样一个，或者只画前N和后N
    # 这里选择：如果 Epoch < 20 全画，否则每隔 (Total/20) 画一个
    unique_epochs = df_batch['Epoch'].unique()
    if len(unique_epochs) > 20:
        step = len(unique_epochs) // 20
        selected_epochs = unique_epochs[::step]
    else:
        selected_epochs = unique_epochs
    
    filtered_batch = df_batch[df_batch['Epoch'].isin(selected_epochs)]
    
    # 修复：添加 hue 参数和 legend=False
    sns.boxplot(x='Epoch', y='Total_Loss', data=filtered_batch, hue='Epoch', palette="Blues", fliersize=1, linewidth=1, legend=False)
    plt.title('Batch Loss Distribution per Epoch (Variance Analysis)')
    plt.xlabel('Epoch')
    plt.ylabel('Batch Loss')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/7_batch_loss_dist.png')
    plt.close()

    # ==========================================
    # [NEW] 图 8: Loss 与 LR 联合分析 (Dual Axis)
    # ==========================================
    fig, ax1_dual = plt.subplots(figsize=(9, 6))
    
    color_loss = '#D76364'
    ax1_dual.set_xlabel('Epoch')
    ax1_dual.set_ylabel('Val Loss', color=color_loss)
    ax1_dual.plot(df_epoch['Epoch'], df_epoch['Val_Loss'], color=color_loss, label='Val Loss', linewidth=2)
    ax1_dual.tick_params(axis='y', labelcolor=color_loss)
    
    ax2_dual = ax1_dual.twinx()  # 实例化第二个轴
    color_lr = '#6D6D6D'
    ax2_dual.set_ylabel('Learning Rate', color=color_lr)
    ax2_dual.plot(df_epoch['Epoch'], df_epoch['LR'], color=color_lr, linestyle='--', alpha=0.6, label='LR')
    ax2_dual.tick_params(axis='y', labelcolor=color_lr)
    ax2_dual.set_yscale('log')

    plt.title('Validation Loss vs Learning Rate')
    fig.tight_layout()
    plt.savefig(f'{SAVE_DIR}/8_loss_lr_combined.png')
    plt.close()

    print("\n🎉 全部 8 张图表生成完毕！已榨干 CSV 的全部潜力！")

if __name__ == '__main__':
    plot_thesis_suite()