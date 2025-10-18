"""
报告生成模块 - 生成HTML可视化报告
Report Generator - Generate HTML visualization reports
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import numpy as np


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'plots')
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 设置绘图样式
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
        plt.rcParams['axes.unicode_minus'] = False
    
    @staticmethod
    def convert_to_json_serializable(obj: Any) -> Any:
        """
        递归转换对象为JSON可序列化格式
        
        Args:
            obj: 输入对象
            
        Returns:
            JSON可序列化的对象
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: ReportGenerator.convert_to_json_serializable(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ReportGenerator.convert_to_json_serializable(item) 
                   for item in obj]
        elif isinstance(obj, tuple):
            return tuple(ReportGenerator.convert_to_json_serializable(item) 
                        for item in obj)
        else:
            return obj
    
    def generate_report(self, analysis_data: Dict, video_path: str, 
                       template_path: str) -> str:
        """
        生成完整的HTML报告
        
        Args:
            analysis_data: 分析数据
            video_path: 视频路径
            template_path: 模板路径
            
        Returns:
            报告文件路径
        """
        print("生成可视化图表...")
        
        # 生成各类图表
        plots = {}
        plots['trunk_tilt'] = self._plot_time_series(
            analysis_data, 'trunk_tilt', '躯干倾斜角度', '角度 (度)')
        
        plots['knee_flexion'] = self._plot_bilateral(
            analysis_data, 'left_knee_flexion', 'right_knee_flexion',
            '膝关节屈曲', '角度 (度)')
        
        plots['ankle_angle'] = self._plot_bilateral(
            analysis_data, 'left_ankle_angle', 'right_ankle_angle',
            '踝关节角度', '角度 (度)')
        
        plots['hip_adduction'] = self._plot_bilateral(
            analysis_data, 'left_hip_adduction', 'right_hip_adduction',
            '髋关节内收', '角度 (度)')
        
        plots['pelvic_drop'] = self._plot_time_series(
            analysis_data, 'pelvic_drop', '骨盆下降', '角度 (度)')
        
        plots['calcaneal'] = self._plot_bilateral(
            analysis_data, 'left_calcaneal_angle', 'right_calcaneal_angle',
            '小腿内外翻', '角度 (度)')
        
        plots['summary'] = self._plot_summary_radar(analysis_data)
        
        print("生成HTML报告...")
        
        # 准备报告数据
        report_data = {
            'video_name': os.path.basename(video_path),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': analysis_data.get('summary', {}),
            'evaluations': analysis_data.get('evaluations', {}),
            'cadence': analysis_data.get('cadence', 0),
            'cadence_rating': analysis_data.get('cadence_rating', 'unknown'),
            'total_frames': analysis_data.get('total_frames', 0),
            'fps': analysis_data.get('fps', 0),
            'plots': plots,
        }
        
        # 读取模板
        with open(template_path, 'r', encoding='utf-8') as f:
            template_str = f.read()
        
        template = Template(template_str)
        html_content = template.render(**report_data)
        
        # 保存HTML报告
        report_path = os.path.join(self.output_dir, 'report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 保存JSON数据（转换numpy数组为列表）
        json_path = os.path.join(self.output_dir, 'analysis_data.json')
        json_serializable_data = self.convert_to_json_serializable(analysis_data)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"报告已生成: {report_path}")
        
        return report_path
    
    def _plot_time_series(self, analysis_data: Dict, field: str, 
                         title: str, ylabel: str) -> str:
        """绘制时间序列图"""
        frame_data = analysis_data.get('frame_data', [])
        
        frames = []
        values = []
        
        for frame in frame_data:
            if frame.get(field) is not None:
                frames.append(frame['frame_idx'])
                values.append(frame[field])
        
        if not values:
            return None
        
        plt.figure(figsize=(12, 5))
        plt.plot(frames, values, linewidth=2, color='#2E86AB')
        plt.fill_between(frames, values, alpha=0.3, color='#2E86AB')
        
        # 添加统计信息
        mean_val = np.mean(values)
        plt.axhline(y=mean_val, color='r', linestyle='--', 
                   label=f'平均值: {mean_val:.1f}°')
        
        plt.xlabel('帧数', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'{field}_timeseries.png'
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'plots/{filename}'
    
    def _plot_bilateral(self, analysis_data: Dict, left_field: str, 
                       right_field: str, title: str, ylabel: str) -> str:
        """绘制双侧对比图"""
        frame_data = analysis_data.get('frame_data', [])
        
        frames = []
        left_values = []
        right_values = []
        
        for frame in frame_data:
            frame_idx = frame['frame_idx']
            left_val = frame.get(left_field)
            right_val = frame.get(right_field)
            
            if left_val is not None or right_val is not None:
                frames.append(frame_idx)
                left_values.append(left_val if left_val is not None else np.nan)
                right_values.append(right_val if right_val is not None else np.nan)
        
        if not frames:
            return None
        
        plt.figure(figsize=(12, 5))
        
        if left_values:
            plt.plot(frames, left_values, linewidth=2, 
                    color='#A23B72', label='左侧', alpha=0.8)
        
        if right_values:
            plt.plot(frames, right_values, linewidth=2, 
                    color='#F18F01', label='右侧', alpha=0.8)
        
        plt.xlabel('帧数', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'{left_field.replace("left_", "")}_bilateral.png'
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'plots/{filename}'
    
    def _plot_summary_radar(self, analysis_data: Dict) -> str:
        """绘制评估雷达图"""
        summary = analysis_data.get('summary', {})
        evaluations = analysis_data.get('evaluations', {})
        
        # 定义评估指标
        categories = [
            ('trunk_tilt', '躯干倾斜'),
            ('knee_flexion', '膝关节屈曲'),
            ('ankle_angle', '踝关节角度'),
            ('hip_adduction', '髋关节内收'),
            ('pelvic_drop', '骨盆下降'),
            ('calcaneal_angle', '小腿角度'),
        ]
        
        labels = []
        scores = []
        
        for key, label in categories:
            # 对于双侧指标，取平均
            if key == 'knee_flexion':
                left_eval = evaluations.get('left_knee_flexion', 'unknown')
                right_eval = evaluations.get('right_knee_flexion', 'unknown')
                eval_score = (self._rating_to_score(left_eval) + 
                            self._rating_to_score(right_eval)) / 2
            elif key == 'ankle_angle':
                left_eval = evaluations.get('left_ankle_angle', 'unknown')
                right_eval = evaluations.get('right_ankle_angle', 'unknown')
                eval_score = (self._rating_to_score(left_eval) + 
                            self._rating_to_score(right_eval)) / 2
            elif key == 'hip_adduction':
                left_eval = evaluations.get('left_hip_adduction', 'unknown')
                right_eval = evaluations.get('right_hip_adduction', 'unknown')
                eval_score = (self._rating_to_score(left_eval) + 
                            self._rating_to_score(right_eval)) / 2
            elif key == 'calcaneal_angle':
                left_eval = evaluations.get('left_calcaneal_angle', 'unknown')
                right_eval = evaluations.get('right_calcaneal_angle', 'unknown')
                eval_score = (self._rating_to_score(left_eval) + 
                            self._rating_to_score(right_eval)) / 2
            else:
                eval_result = evaluations.get(key, 'unknown')
                eval_score = self._rating_to_score(eval_result)
            
            labels.append(label)
            scores.append(eval_score)
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        scores += scores[:1]  # 闭合
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, scores, 'o-', linewidth=2, color='#2E86AB')
        ax.fill(angles, scores, alpha=0.25, color='#2E86AB')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=9)
        ax.set_title('跑步姿势综合评估', size=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        
        filename = 'summary_radar.png'
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'plots/{filename}'
    
    def _rating_to_score(self, rating: str) -> float:
        """将评级转换为分数"""
        rating_map = {
            'good': 100,
            'mediocre': 60,
            'bad': 30,
            'unknown': 0
        }
        return rating_map.get(rating, 0)
    
    def _score_to_rating(self, score: float) -> str:
        """将分数转换为评级"""
        if score >= 80:
            return 'good'
        elif score >= 50:
            return 'mediocre'
        else:
            return 'bad'

