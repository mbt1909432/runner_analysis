# 快速开始指南 🚀

## 第一步：安装依赖

### 1. 安装 PyTorch (CUDA 12.6)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 2. 安装其他依赖

```bash
pip install -r requirements.txt
```

## 第二步：测试GPU

```bash
python test_gpu.py
```

如果看到 "✓ 所有测试通过!"，说明环境配置成功！

## 第三步：运行分析

### 基本用法

```bash
python main.py --video your_running_video.mp4
```

### 查看示例

```bash
python example_usage.py
```

## 输出结果

分析完成后，在 `output` 目录中会生成：

- `report.html` - 可视化报告（在浏览器中打开查看）
- `annotated_video.mp4` - 标注后的视频
- `analysis_data.json` - 原始分析数据
- `plots/` - 各项指标的图表

## 常见问题

### Q: CUDA不可用怎么办？

A: 可以使用CPU模式（会比较慢）：
```bash
python main.py --video video.mp4 --device cpu
```

### Q: 如何获得更快的处理速度？

A: 使用轻量级模型：
```bash
python main.py --video video.mp4 --model yolov8n-pose.pt
```

### Q: 如何获得更高的精度？

A: 使用大型模型：
```bash
python main.py --video video.mp4 --model yolov8x-pose.pt
```

## 视频要求

- **格式**: MP4, AVI, MOV等常见格式
- **质量**: 1080p或更高（推荐）
- **帧率**: 30 FPS或更高
- **拍摄角度**: 侧面拍摄效果最佳
- **光照**: 光线充足，人物清晰可见

## 需要帮助？

查看完整文档：`README.md`

运行示例命令：`python example_usage.py --check`

