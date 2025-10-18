@echo off
echo ========================================
echo 跑步姿势分析系统 - 安装脚本
echo ========================================
echo.

echo [1/3] 安装 PyTorch (CUDA 12.6)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
echo.

echo [2/3] 安装其他依赖...
pip install -r requirements.txt
echo.

echo [3/3] 测试GPU...
python test_gpu.py
echo.

echo ========================================
echo 安装完成!
echo ========================================
echo.
echo 快速开始:
echo   python main.py --video your_video.mp4
echo.
echo 查看使用示例:
echo   python example_usage.py
echo.

pause

