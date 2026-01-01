import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import FaceAligner
from config import Config
import shutil

def process_dataset(source_root, target_root, aligner, img_size=224, dataset_type='flat'):
    """
    清洗并处理数据集
    dataset_type: 'flat' (UTKFace/AAF: 所有图片在一个文件夹) 
                  'nested' (AFAD: age/gender/xxx.jpg)
    """
    if not os.path.exists(source_root):
        print(f"⚠️ 源目录不存在: {source_root}")
        return

    print(f"🔄 开始处理: {source_root} -> {target_root}", flush=True)
    if not os.path.exists(target_root):
        os.makedirs(target_root)
        
    failed_count = 0
    success_count = 0
    fallback_count = 0
    
    if dataset_type == 'flat':
        files = [f for f in os.listdir(source_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"📂 找到 {len(files)} 张图片", flush=True)
        
        for i, f in enumerate(files):
            if i % 100 == 0:
                print(f"Processing {i}/{len(files)}: {f}", flush=True)
                
            src_path = os.path.join(source_root, f)
            dst_path = os.path.join(target_root, f)
            
            try:
                # 1. Load
                image = Image.open(src_path).convert('RGB')
                
                aligned_img = None
                # 2. Try Align
                if aligner:
                    aligned_img = aligner.align(image, desired_size=img_size)
                
                # 3. Fallback to Resize if Align Failed or Aligner disabled
                if aligned_img is None:
                    if aligner: 
                        fallback_count += 1
                        # print(f"⚠️ Align failed for {f}, using resize fallback.")
                    
                    # 使用 Center Crop resize 保持比例
                    # 先缩放到短边为 img_size
                    w, h = image.size
                    scale = img_size / min(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = image.resize((new_w, new_h), Image.BILINEAR)
                    
                    # Crop Center
                    left = (new_w - img_size) // 2
                    top = (new_h - img_size) // 2
                    aligned_img = image.crop((left, top, left + img_size, top + img_size))

                # 4. Save
                aligned_img.save(dst_path, quality=95)
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error processing {f}: {e}", flush=True)
                failed_count += 1
                
    elif dataset_type == 'nested':
        # 用于 AFAD
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

    print(f"✅ 完成! 总成功: {success_count} (Fallback: {fallback_count}), ❌失败: {failed_count}", flush=True)

def main():
    print("🚀 启动预处理脚本 (带人脸对齐 & Fallback)...", flush=True)
    cfg = Config()
    
    # Enable FaceAligner
    aligner = FaceAligner()
    print("✅ FaceAligner initialized.", flush=True)
    
    # 定义新的根目录
    base_aligned_dir = "./data_aligned"
    
    # 1. UTKFace Train
    raw_train_dir = "./data/UTKFace/train"
    print(f"Checking {raw_train_dir}...", flush=True)
    process_dataset(raw_train_dir, 
                    os.path.join(base_aligned_dir, "UTKFace", "train"), 
                    aligner, cfg.img_size, 'flat')
                    
    # 2. UTKFace Val
    raw_val_dir = "./data/UTKFace/val"
    print(f"Checking {raw_val_dir}...", flush=True)
    process_dataset(raw_val_dir, 
                    os.path.join(base_aligned_dir, "UTKFace", "val"), 
                    aligner, cfg.img_size, 'flat')

    # 3. AFAD
    raw_afad_dir = r"F:\QQFiles\Study\shit\tarball\tarball-master\AFAD-Full.tar\AFAD-Full~\AFAD-Full"
    print(f"Checking AFAD...", flush=True)
    if os.path.exists(raw_afad_dir):
        process_dataset(raw_afad_dir, 
                        os.path.join(base_aligned_dir, "AFAD"), 
                        aligner, cfg.img_size, 'nested')
    else:
        print(f"⚠️ 未找到 AFAD 源目录: {raw_afad_dir}", flush=True)

    # 4. AAF
    raw_aaf_dir = r"F:\QQFiles\Study\shit\AAF\All-Age-Faces Dataset\aglined_faces"
    print(f"Checking AAF...", flush=True)
    if os.path.exists(raw_aaf_dir):
        process_dataset(raw_aaf_dir, 
                        os.path.join(base_aligned_dir, "AAF"), 
                        aligner, cfg.img_size, 'flat')
    else:
        print(f"⚠️ 未找到 AAF 源目录: {raw_aaf_dir}", flush=True)
                        
    print("\n🎉 数据预处理全部完成！新数据位于 ./data_aligned", flush=True)
    print("请记得更新 config.py 中的路径！", flush=True)

if __name__ == "__main__":
    main()
