"""
使用示例脚本 - 演示如何使用跑步姿势分析系统
Example Usage Script
"""

import os
import sys


def print_examples():
    """打印使用示例"""
    print("=" * 70)
    print("跑步姿势分析系统 - 使用示例")
    print("=" * 70)
    print()
    
    print("1. 基本使用 (使用默认配置)")
    print("-" * 70)
    print("python main.py --video your_running_video.mp4")
    print()
    
    print("2. 指定输出目录")
    print("-" * 70)
    print("python main.py --video running.mp4 --output output/my_analysis")
    print()
    
    print("3. 使用不同的模型")
    print("-" * 70)
    print("# 使用轻量级模型 (更快，精度稍低)")
    print("python main.py --video running.mp4 --model yolov8n-pose.pt")
    print()
    print("# 使用高精度模型 (更慢，精度更高)")
    print("python main.py --video running.mp4 --model yolov8x-pose.pt")
    print()
    
    print("4. 调整检测置信度")
    print("-" * 70)
    print("python main.py --video running.mp4 --confidence 0.7")
    print()
    
    print("5. 实时显示处理过程")
    print("-" * 70)
    print("python main.py --video running.mp4 --show-video")
    print()
    
    print("6. 使用CPU模式 (无GPU时)")
    print("-" * 70)
    print("python main.py --video running.mp4 --device cpu")
    print()
    
    print("7. 完整示例 (所有参数)")
    print("-" * 70)
    print("python main.py \\")
    print("    --video input/running.mp4 \\")
    print("    --output output/analysis_2024 \\")
    print("    --model yolov8x-pose.pt \\")
    print("    --confidence 0.6 \\")
    print("    --device cuda \\")
    print("    --show-video")
    print()
    
    print("=" * 70)
    print("可用的YOLO模型:")
    print("-" * 70)
    print("  yolov8n-pose.pt  - Nano    (最快，精度最低，适合快速测试)")
    print("  yolov8s-pose.pt  - Small   (快速，精度较低)")
    print("  yolov8m-pose.pt  - Medium  (平衡速度和精度)")
    print("  yolov8l-pose.pt  - Large   (较慢，精度较高)")
    print("  yolov8x-pose.pt  - XLarge  (最慢，精度最高，推荐用于最终分析)")
    print()
    print("注意: 首次使用时会自动下载模型文件")
    print("=" * 70)


def check_environment():
    """检查环境配置"""
    print("\n检查环境配置...")
    print("-" * 70)
    
    # 检查Python版本
    import sys
    print(f"✓ Python版本: {sys.version.split()[0]}")
    
    # 检查必要的包
    packages = {
        'torch': 'PyTorch',
        'ultralytics': 'Ultralytics (YOLO)',
        'cv2': 'OpenCV',
        'yaml': 'PyYAML',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'jinja2': 'Jinja2'
    }
    
    missing_packages = []
    
    for package, name in packages.items():
        try:
            if package == 'cv2':
                import cv2
                print(f"✓ {name}: {cv2.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: 未安装")
            missing_packages.append(name)
    
    if missing_packages:
        print("\n缺少以下包，请运行:")
        print("pip install -r requirements.txt")
        return False
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA: 可用 (版本 {torch.version.cuda})")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA: 不可用 (将使用CPU模式)")
    except:
        pass
    
    # 检查配置文件
    if os.path.exists('config.yaml'):
        print("✓ 配置文件: config.yaml")
    else:
        print("✗ 配置文件: config.yaml 不存在")
        return False
    
    # 检查模板文件
    if os.path.exists('templates/report_template.html'):
        print("✓ 报告模板: templates/report_template.html")
    else:
        print("✗ 报告模板: templates/report_template.html 不存在")
        return False
    
    print("-" * 70)
    print("✓ 环境检查完成!\n")
    return True


def create_sample_command():
    """生成示例命令"""
    print("\n快速开始:")
    print("=" * 70)
    print("如果您有一个跑步视频 'my_running.mp4'，运行:")
    print()
    print("    python main.py --video my_running.mp4")
    print()
    print("分析结果将保存在 'output' 目录中")
    print("=" * 70)


if __name__ == '__main__':
    print_examples()
    
    if '--check' in sys.argv or '-c' in sys.argv:
        check_environment()
    
    create_sample_command()
    
    print("\n提示:")
    print("  - 运行 'python test_gpu.py' 测试GPU是否可用")
    print("  - 运行 'python example_usage.py --check' 检查环境配置")
    print("  - 查看 README.md 了解更多信息")
    print()

