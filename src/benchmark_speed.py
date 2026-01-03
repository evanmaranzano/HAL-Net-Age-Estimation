import torch
import time
import numpy as np
import psutil
import torch.backends.cudnn as cudnn
from model import LightweightAgeEstimator
from config import Config

def benchmark_device(device_name, num_iterations=1000, batch_size=1):
    print(f"\n🚀 Benchmarking on {device_name} (Batch Size: {batch_size})...")
    
    # 1. Setup Model
    cfg = Config()
    device = torch.device(device_name)
    model = LightweightAgeEstimator(cfg).to(device)
    model.eval()
    
    # Enable cuDNN benchmark for GPU
    if device.type == 'cuda':
        cudnn.benchmark = True
        
    # 2. Fake Data
    input_shape = (batch_size, 3, cfg.img_size, cfg.img_size)
    dummy_input = torch.randn(input_shape).to(device)
    
    # 3. Warm-up
    print("  🔥 Warming up...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
    
    # Synchronize for GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # 4. Measure
    print(f"  ⏱️ Running {num_iterations} iterations...")
    timings = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            timings.append(end - start)
            
    # 5. Stats
    timings = np.array(timings)
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    fps = batch_size / avg_time
    
    print(f"  ✅ Result:")
    print(f"     Latency: {avg_time*1000:.2f} ms ± {std_time*1000:.2f} ms")
    print(f"     Throughput: {fps:.2f} FPS")
    
    return avg_time, fps

def main():
    print("="*60)
    print("🏁 MobileNetV3 Age Estimation Speed Benchmark")
    print("="*60)
    
    # CPU info
    print(f"🖥️ CPU Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"🧠 Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected!")
        
    print("-" * 60)
    
    # --- CPU Benchmark ---
    cpu_latency, cpu_fps = benchmark_device('cpu', num_iterations=200)
    
    # --- GPU Benchmark ---
    if torch.cuda.is_available():
        gpu_latency, gpu_fps = benchmark_device('cuda', num_iterations=1000)
    
    # --- Report Summary ---
    print("\n" + "="*60)
    print("📊 Final Report")
    print("="*60)
    print(f"CPU Inference (Ryzen 9 6900HX):  {cpu_fps:.1f} FPS | {cpu_latency*1000:.1f} ms")
    if torch.cuda.is_available():
        print(f"GPU Inference (RTX 3060 Laptop): {gpu_fps:.1f} FPS | {gpu_latency*1000:.1f} ms")
        print(f"Speedup Factors: {gpu_fps/cpu_fps:.1f}x faster on GPU")

if __name__ == "__main__":
    main()
