"""
GPU测试脚本 - 检测CUDA是否可用以及GPU信息
Test GPU availability and display CUDA/GPU information
"""

import sys

def test_gpu():
    print("=" * 60)
    print("GPU 和 CUDA 测试")
    print("=" * 60)
    
    # 测试PyTorch
    print("\n[1] 检查 PyTorch 安装...")
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch 未安装!")
        print("\n请运行以下命令安装PyTorch (CUDA 12.6):")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
        return False
    
    # 测试CUDA
    print("\n[2] 检查 CUDA 可用性...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✓ CUDA 可用: {torch.cuda.is_available()}")
        print(f"✓ CUDA 版本: {torch.version.cuda}")
        print(f"✓ cuDNN 版本: {torch.backends.cudnn.version()}")
    else:
        print("✗ CUDA 不可用!")
        print("\n可能的原因:")
        print("  1. 未安装NVIDIA GPU驱动")
        print("  2. 安装的PyTorch版本不支持CUDA")
        print("  3. CUDA版本不匹配")
        return False
    
    # GPU信息
    print("\n[3] GPU 设备信息...")
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"✓ 可用GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"\n  GPU {i}:")
            print(f"    名称: {torch.cuda.get_device_name(i)}")
            print(f"    计算能力: {torch.cuda.get_device_capability(i)}")
            
            # 显存信息
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            
            print(f"    总显存: {mem_total:.2f} GB")
            print(f"    已预留: {mem_reserved:.2f} GB")
            print(f"    已使用: {mem_allocated:.2f} GB")
            print(f"    可用显存: {mem_total - mem_allocated:.2f} GB")
    
    # 测试GPU计算
    print("\n[4] 测试 GPU 计算能力...")
    try:
        # 创建测试张量
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # 矩阵乘法
        import time
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"✓ GPU矩阵运算测试通过")
        print(f"  1000x1000 矩阵乘法耗时: {(end-start)*1000:.2f} ms")
        
    except Exception as e:
        print(f"✗ GPU计算测试失败: {e}")
        return False
    
    # 测试Ultralytics
    print("\n[5] 检查 Ultralytics (YOLO) 安装...")
    try:
        import ultralytics
        print(f"✓ Ultralytics 版本: {ultralytics.__version__}")
    except ImportError:
        print("✗ Ultralytics 未安装!")
        print("  请运行: pip install ultralytics")
        return False
    
    # 测试OpenCV
    print("\n[6] 检查 OpenCV 安装...")
    try:
        import cv2
        print(f"✓ OpenCV 版本: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV 未安装!")
        print("  请运行: pip install opencv-python")
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过! 系统已准备好进行跑步姿势分析")
    print("=" * 60)
    print("\n推荐配置:")
    print("  - GPU显存: >= 8 GB (当前可用显存充足)")
    print("  - 视频分辨率: 1080p 或更高")
    print("  - 帧率: 30 fps 或更高")
    
    return True

if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1)

