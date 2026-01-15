import os
import sys

# Add project root to path to allow imports from src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.utils import FaceAligner
from src.config import Config
import shutil

def process_afad(source_root, target_root, aligner, img_size=224):
    """
    处理 AFAD 数据集 (nested structure)
    """
    if not os.path.exists(source_root):
        print(f"⚠️ AFAD 源目录不存在: {source_root}")
        return

    print(f"🔄 开始处理 AFAD: {source_root} -> {target_root}", flush=True)
    if not os.path.exists(target_root):
        os.makedirs(target_root)
        
    failed_count = 0
    success_count = 0
    fallback_count = 0
    
    all_folders = os.listdir(source_root)
    print(f"📂 找到 {len(all_folders)} 个年龄文件夹", flush=True)
    
    for i, age_folder in enumerate(all_folders):
        if i % 10 == 0:
            print(f"Processing Age Folder {i}/{len(all_folders)}: {age_folder}", flush=True)
        age_src = os.path.join(source_root, age_folder)
        age_dst = os.path.join(target_root, age_folder)
        
        if not os.path.isdir(age_src): continue
        
        for gender_folder in os.listdir(age_src):
            gender_src = os.path.join(age_src, gender_folder)
            gender_dst = os.path.join(age_dst, gender_folder)
            
            if not os.path.isdir(gender_src): continue
            
            if not os.path.exists(gender_dst):
                os.makedirs(gender_dst)
            
            images = [x for x in os.listdir(gender_src) if x.lower().endswith(('.jpg', '.png', '.jpeg'))]
            for f in images:
                src_path = os.path.join(gender_src, f)
                dst_path = os.path.join(gender_dst, f)
                
                try:
                    image = Image.open(src_path).convert('RGB')
                    
                    aligned_img = None
                    if aligner:
                        aligned_img = aligner.align(image, desired_size=img_size)
                    
                    if aligned_img is None:
                        if aligner: fallback_count += 1
                        w, h = image.size
                        scale = img_size / min(w, h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        image = image.resize((new_w, new_h), Image.BILINEAR)
                        left = (new_w - img_size) // 2
                        top = (new_h - img_size) // 2
                        aligned_img = image.crop((left, top, left + img_size, top + img_size))

                    aligned_img.save(dst_path, quality=95)
                    success_count += 1
                except Exception as e:
                    print(f"Error {f}: {e}", flush=True)
                    failed_count += 1

    print(f"✅ AFAD 完成! 总成功: {success_count} (Fallback: {fallback_count}), ❌失败: {failed_count}", flush=True)

def main():
    print("🚀 启动预处理脚本 (带人脸对齐 & Fallback)...", flush=True)
    cfg = Config()
    
    # Enable FaceAligner
    aligner = FaceAligner()
    print("✅ FaceAligner initialized.", flush=True)
    
    # 定义新的根目录
    base_aligned_dir = os.path.join(PROJECT_ROOT, "datasets")
    
    # Process Only AFAD
    raw_afad_dir = r"F:\QQFiles\Study\shit\tarball\tarball-master\AFAD-Full.tar\AFAD-Full~\AFAD-Full"
    print(f"Checking AFAD...", flush=True)
    if os.path.exists(raw_afad_dir):
        process_afad(raw_afad_dir, 
                        os.path.join(base_aligned_dir, "AFAD"), 
                        aligner, cfg.img_size)
    else:
        print(f"⚠️ 未找到 AFAD 源目录: {raw_afad_dir}", flush=True)
                        
    print(f"\n🎉 数据预处理全部完成！新数据位于 {base_aligned_dir}", flush=True)
    print("请记得更新 config.py 中的路径！", flush=True)

if __name__ == "__main__":
    main()
