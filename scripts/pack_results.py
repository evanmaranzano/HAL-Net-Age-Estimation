import os
import zipfile
import datetime
import glob

def pack_results():
    # Define root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Generate timestamped filename
    zip_filename = os.path.join(root_dir, "training_results.zip")
    
    print(f"📦 Start packaging to: {zip_filename}")
    
    # Files/Dirs to include
    patterns = [
        "*.csv",                # Logs
        "*.txt",                # Final results
        "best_model_*.pth",     # Best weights
        "swa_model_*.pth",      # SWA weights
        "last_checkpoint*.pth", # Last state
        "checkpoint_seed*.pth", # Last 10 epochs (SWA candidates)
        "runs",                 # TensorBoard
        "plots",                # Plots
        "src",                  # Source code (for reproducibility)
        "*.docx"                # Report files
    ]
    
    count = 0
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Walk through patterns in ROOT
        for pattern in patterns:
            full_pattern = os.path.join(root_dir, pattern)
            # Handle recursive dirs vs files
            if pattern in ["runs", "plots", "src"]:
                target_dir = os.path.join(root_dir, pattern)
                if os.path.exists(target_dir):
                    for root, dirs, files in os.walk(target_dir):
                        # Skip __pycache__
                        if "__pycache__" in root:
                            continue
                            
                        for file in files:
                            if file.endswith(".pyc") or file == ".DS_Store":
                                continue
                                
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, root_dir)
                            print(f"  Adding: {arcname}")
                            zipf.write(file_path, arcname)
                            count += 1
            else:
                # Glob files in root
                for file_path in glob.glob(full_pattern):
                    arcname = os.path.relpath(file_path, root_dir)
                    print(f"  Adding: {arcname}")
                    zipf.write(file_path, arcname)
                    count += 1
                    
    print(f"\n✅ Packaging complete!")
    print(f"📁 Total files: {count}")
    print(f"💾 Size: {os.path.getsize(zip_filename) / 1024 / 1024:.2f} MB")
    print(f"📍 Location: {zip_filename}")

if __name__ == "__main__":
    pack_results()
