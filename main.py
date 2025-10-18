"""
主程序 - 跑步姿势分析系统
Main Program - Running Pose Analysis System
"""

import argparse
import os
import sys
import yaml
import cv2
import torch
from tqdm import tqdm
from src.detector import PoseDetector
from src.analyzer import RunningPoseAnalyzer
from src.report_generator import ReportGenerator


def load_config(config_path: str = 'config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def process_video(video_path: str, output_dir: str, config: dict, 
                 show_video: bool = False):
    """
    处理视频并生成分析报告
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        config: 配置字典
        show_video: 是否显示处理过程
    """
    # 检查视频文件
    if not os.path.exists(video_path):
        print(f"错误: 找不到视频文件 '{video_path}'")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("跑步姿势分析系统")
    print("=" * 60)
    print(f"输入视频: {video_path}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 初始化检测器
    print("[1/5] 初始化姿态检测器...")
    detector = PoseDetector(
        model_name=config['model']['name'],
        confidence=config['model']['confidence'],
        device=config['model']['device']
    )
    
    # 初始化分析器
    print("[2/5] 初始化姿势分析器...")
    analyzer = RunningPoseAnalyzer(config)
    
    # 打开视频
    print("[3/5] 处理视频...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}'")
        return False
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  视频信息: {frame_width}x{frame_height} @ {fps} FPS")
    print(f"  总帧数: {total_frames}")
    
    # 创建视频写入器
    output_video_path = os.path.join(output_dir, 'annotated_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                          (frame_width, frame_height))
    
    # 处理每一帧
    frame_idx = 0
    skip_frames = config['video'].get('skip_frames', 1)
    
    pbar = tqdm(total=total_frames, desc="  处理进度")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 跳帧处理
        if frame_idx % skip_frames == 0:
            # 检测姿态
            detections = detector.detect(frame)
            
            # 分析姿势
            if detections:
                # 取第一个检测到的人
                analyzer.analyze_frame(detections[0], frame_idx)
                
                # 绘制姿态
                annotated = detector.draw_pose(
                    frame, detections, 
                    show_confidence=config['visualization']['show_confidence']
                )
            else:
                annotated = frame.copy()
            
            # 写入输出视频
            out.write(annotated)
            
            # 显示处理过程
            if show_video:
                cv2.imshow('Running Pose Analysis', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户中断处理")
                    break
        else:
            out.write(frame)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    if show_video:
        cv2.destroyAllWindows()
    
    print(f"  标注视频已保存: {output_video_path}")
    
    # 平滑数据
    print("[4/5] 平滑和分析数据...")
    analyzer.smooth_data()
    
    # 计算统计数据
    summary = analyzer.get_summary_statistics()
    
    # 计算步频
    cadence = analyzer.calculate_cadence(fps)
    cadence_rating = analyzer.evaluate_cadence(cadence)
    
    print(f"  步频: {cadence:.1f} 步/分钟 ({cadence_rating})")
    
    # 评估各项指标
    evaluations = {}
    for field in ['trunk_tilt', 'left_knee_flexion', 'right_knee_flexion',
                  'left_ankle_angle', 'right_ankle_angle',
                  'left_hip_adduction', 'right_hip_adduction',
                  'pelvic_drop', 'left_calcaneal_angle', 'right_calcaneal_angle']:
        if summary.get(field):
            mean_value = summary[field]['mean']
            rating = analyzer.evaluate_parameter(field, mean_value)
            evaluations[field] = rating
            
            # 打印评估结果
            field_name_cn = {
                'trunk_tilt': '躯干倾斜',
                'left_knee_flexion': '左膝屈曲',
                'right_knee_flexion': '右膝屈曲',
                'left_ankle_angle': '左踝角度',
                'right_ankle_angle': '右踝角度',
                'left_hip_adduction': '左髋内收',
                'right_hip_adduction': '右髋内收',
                'pelvic_drop': '骨盆下降',
                'left_calcaneal_angle': '左小腿角度',
                'right_calcaneal_angle': '右小腿角度',
            }
            
            rating_cn = {'good': '良好', 'mediocre': '中等', 'bad': '不佳', 'unknown': '未知'}
            
            print(f"  {field_name_cn.get(field, field)}: {mean_value:.1f}° ({rating_cn.get(rating, rating)})")
    
    # 准备报告数据
    analysis_data = {
        'summary': summary,
        'evaluations': evaluations,
        'cadence': cadence,
        'cadence_rating': cadence_rating,
        'frame_data': analyzer.frame_data,
        'total_frames': total_frames,
        'fps': fps,
        'video_info': {
            'width': frame_width,
            'height': frame_height,
            'duration': total_frames / fps
        }
    }
    
    # 生成报告
    print("[5/5] 生成分析报告...")
    report_generator = ReportGenerator(output_dir)
    
    template_path = 'templates/report_template.html'
    report_path = report_generator.generate_report(
        analysis_data, video_path, template_path
    )
    
    print()
    print("=" * 60)
    print("✓ 分析完成!")
    print(f"  报告: {report_path}")
    print(f"  视频: {output_video_path}")
    print(f"  数据: {os.path.join(output_dir, 'analysis_data.json')}")
    print("=" * 60)
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='跑步姿势分析系统 - 基于YOLOv8姿态估计'
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='输入视频路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='输出目录 (默认: output)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='配置文件路径 (默认: config.yaml)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='YOLO模型路径 (覆盖配置文件)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=None,
        help='检测置信度阈值 (覆盖配置文件)'
    )
    
    parser.add_argument(
        '--show-video',
        action='store_true',
        help='实时显示处理过程'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='运行设备 (覆盖配置文件)'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"错误: 无法加载配置文件: {e}")
        return 1
    
    # 覆盖配置
    if args.model:
        config['model']['name'] = args.model
    
    if args.confidence:
        config['model']['confidence'] = args.confidence
    
    if args.device:
        config['model']['device'] = args.device
    
    # 检查CUDA可用性
    if config['model']['device'] == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU模式")
        print("提示: 运行 'python test_gpu.py' 检查GPU状态")
        config['model']['device'] = 'cpu'
    
    # 处理视频
    try:
        success = process_video(
            args.video,
            args.output,
            config,
            args.show_video
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
        return 1
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

