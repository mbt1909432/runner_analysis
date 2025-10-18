# 跑步姿势分析系统 (Running Pose Analysis System)

基于YOLOv8/v11姿态估计的跑步姿势分析系统，可以分析视频中的跑步姿势并生成详细的评估报告。

## 功能特点

- 🏃 实时跑步姿势检测和分析
- 📊 多维度姿势评估（7项指标）
- 📈 详细的HTML可视化报告
- 🎥 支持MP4视频输入
- ⚡ GPU加速（CUDA 12.6支持）

## 分析指标

1. **前后躯干倾斜** (Anterior/Posterior Trunk Tilt)
2. **膝关节屈曲** (Knee Flexion)
3. **踝关节跖屈** (Ankle Plantar)
4. **髋关节内收** (Hip Adduction)
5. **骨盆下降** (Pelvic Drop)
6. **小腿内外翻** (Calcaneal Inversion/Eversion)
7. **步频** (Cadence)

## 安装步骤

### 1. 安装PyTorch（CUDA 12.6版本）

```bash
# Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 2. 安装其他依赖

```bash
pip install -r requirements.txt
```

### 3. 测试GPU是否可用

```bash
python test_gpu.py
```

## 快速开始

### 基本使用

```bash
python main.py --video path/to/your/running_video.mp4 --output output/report
```
python main.py --video E:\pycharm_project\mmpose2\test_running.mp4 --output output/report2
python main.py --video E:\pycharm_project\mmpose2\test_running2.mp4 --output output/report
### 高级选项

```bash
python main.py \
    --video input/running.mp4 \
    --output output/analysis \
    --model yolov8x-pose.pt \
    --confidence 0.5 \
    --show-video
```

## 参数说明

- `--video`: 输入视频路径（必需）
- `--output`: 输出目录（默认：output）
- `--model`: YOLO模型路径（默认：yolov8x-pose.pt）
- `--confidence`: 检测置信度阈值（默认：0.5）
- `--show-video`: 实时显示处理过程
- `--fps`: 采样帧率（默认：使用原视频帧率）

## 输出内容

- `report.html`: 详细的可视化分析报告
- `analysis_data.json`: 原始分析数据
- `annotated_video.mp4`: 标注后的视频
- `plots/`: 各项指标的可视化图表

## 评估标准

根据运动医学标准对跑步姿势进行评级：

- ✅ **良好 (Good)**: 姿势标准，风险低
- ⚠️ **中等 (Mediocre)**: 姿势可改进，中等风险
- ❌ **不佳 (Bad)**: 姿势需要纠正，高风险

## 系统要求

- Python 3.8+
- NVIDIA GPU with CUDA 12.6
- 至少8GB GPU内存（推荐）
- Windows 10/11

## 项目结构

```
mmpose2/
├── main.py                 # 主程序
├── test_gpu.py            # GPU测试脚本
├── config.yaml            # 配置文件
├── requirements.txt       # 依赖列表
├── src/
│   ├── detector.py        # 姿态检测
│   ├── analyzer.py        # 姿势分析
│   ├── angle_calculator.py # 角度计算
│   └── report_generator.py # 报告生成
└── templates/
    └── report_template.html # 报告模板
```

## 注意事项

1. 确保视频中跑步者清晰可见
2. 建议从侧面拍摄以获得最佳分析效果
3. 视频质量越高，分析精度越好
4. 首次运行会自动下载YOLO模型（约100MB）

## License

MIT License

