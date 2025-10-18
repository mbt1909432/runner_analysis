"""
跑步姿势分析模块 - 分析跑步姿势并评估各项指标
Running Pose Analyzer - Analyze running pose and evaluate metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d
from .angle_calculator import AngleCalculator


class RunningPoseAnalyzer:
    """跑步姿势分析器"""
    
    def __init__(self, config: Dict):
        """
        初始化分析器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.thresholds = config['thresholds']
        self.smoothing = config.get('smoothing', {'enabled': True, 'window_size': 5})
        self.gait_config = config.get('gait', {'min_stride_frames': 10, 'foot_speed_threshold': 2.0})
        
        self.angle_calc = AngleCalculator()
        
        # 存储帧级别的数据
        self.frame_data = []
        
    def analyze_frame(self, detection: Dict, frame_idx: int) -> Dict:
        """
        分析单帧的跑步姿势
        
        Args:
            detection: 姿态检测结果
            frame_idx: 帧索引
            
        Returns:
            分析结果
        """
        if not detection or 'keypoints' not in detection:
            return None
        
        keypoints = detection['keypoints']
        
        # 提取关键点
        nose = self._get_kpt(keypoints, 'nose')
        left_shoulder = self._get_kpt(keypoints, 'left_shoulder')
        right_shoulder = self._get_kpt(keypoints, 'right_shoulder')
        left_hip = self._get_kpt(keypoints, 'left_hip')
        right_hip = self._get_kpt(keypoints, 'right_hip')
        left_knee = self._get_kpt(keypoints, 'left_knee')
        right_knee = self._get_kpt(keypoints, 'right_knee')
        left_ankle = self._get_kpt(keypoints, 'left_ankle')
        right_ankle = self._get_kpt(keypoints, 'right_ankle')
        
        # 初始化结果
        result = {
            'frame_idx': frame_idx,
            'trunk_tilt': None,
            'left_knee_flexion': None,
            'right_knee_flexion': None,
            'left_ankle_angle': None,
            'right_ankle_angle': None,
            'left_hip_adduction': None,
            'right_hip_adduction': None,
            'pelvic_drop': None,
            'left_calcaneal_angle': None,
            'right_calcaneal_angle': None,
            'left_foot_position': None,
            'right_foot_position': None,
        }
        
        # 1. 计算躯干倾斜
        if left_shoulder is not None and right_shoulder is not None and \
           left_hip is not None and right_hip is not None:
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            result['trunk_tilt'] = self.angle_calc.calculate_trunk_tilt(shoulder_center, hip_center)
        
        # 2. 计算膝关节屈曲 - 左侧
        if left_hip is not None and left_knee is not None and left_ankle is not None:
            result['left_knee_flexion'] = self.angle_calc.calculate_knee_flexion(
                left_hip, left_knee, left_ankle)
        
        # 3. 计算膝关节屈曲 - 右侧
        if right_hip is not None and right_knee is not None and right_ankle is not None:
            result['right_knee_flexion'] = self.angle_calc.calculate_knee_flexion(
                right_hip, right_knee, right_ankle)
        
        # 4. 计算踝关节角度 - 左侧
        if left_knee is not None and left_ankle is not None:
            # 使用脚趾估计点（踝关节下方）
            foot_estimate = left_ankle + np.array([0, 30])
            result['left_ankle_angle'] = self.angle_calc.calculate_ankle_angle(
                left_knee, left_ankle, foot_estimate)
        
        # 5. 计算踝关节角度 - 右侧
        if right_knee is not None and right_ankle is not None:
            foot_estimate = right_ankle + np.array([0, 30])
            result['right_ankle_angle'] = self.angle_calc.calculate_ankle_angle(
                right_knee, right_ankle, foot_estimate)
        
        # 6. 计算髋关节内收 - 左侧
        if left_hip is not None and right_hip is not None and left_knee is not None and left_ankle is not None:
            result['left_hip_adduction'] = self.angle_calc.calculate_hip_adduction(
                left_hip, right_hip, left_knee, left_ankle)
        
        # 7. 计算髋关节内收 - 右侧
        if left_hip is not None and right_hip is not None and right_knee is not None and right_ankle is not None:
            result['right_hip_adduction'] = self.angle_calc.calculate_hip_adduction(
                left_hip, right_hip, right_knee, right_ankle)
        
        # 8. 计算骨盆下降
        if left_hip is not None and right_hip is not None:
            result['pelvic_drop'] = self.angle_calc.calculate_pelvic_drop(left_hip, right_hip)
        
        # 9. 计算小腿内外翻 - 左侧
        if left_knee is not None and left_ankle is not None:
            result['left_calcaneal_angle'] = self.angle_calc.calculate_calcaneal_angle(
                left_knee, left_ankle)
        
        # 10. 计算小腿内外翻 - 右侧
        if right_knee is not None and right_ankle is not None:
            result['right_calcaneal_angle'] = self.angle_calc.calculate_calcaneal_angle(
                right_knee, right_ankle)
        
        # 记录脚部位置（用于步频计算）
        if left_ankle is not None:
            result['left_foot_position'] = left_ankle.copy()
        if right_ankle is not None:
            result['right_foot_position'] = right_ankle.copy()
        
        self.frame_data.append(result)
        
        return result
    
    def _get_kpt(self, keypoints: Dict, name: str) -> Optional[np.ndarray]:
        """获取关键点坐标"""
        if name not in keypoints:
            return None
        kpt = keypoints[name]
        if kpt['confidence'] < 0.3:
            return None
        return np.array([kpt['x'], kpt['y']])
    
    def smooth_data(self) -> None:
        """平滑时间序列数据"""
        if not self.smoothing['enabled'] or len(self.frame_data) < 5:
            return
        
        window_size = self.smoothing.get('window_size', 5)
        method = self.smoothing.get('method', 'savgol')
        
        # 需要平滑的字段
        fields_to_smooth = [
            'trunk_tilt', 'left_knee_flexion', 'right_knee_flexion',
            'left_ankle_angle', 'right_ankle_angle',
            'left_hip_adduction', 'right_hip_adduction',
            'pelvic_drop', 'left_calcaneal_angle', 'right_calcaneal_angle'
        ]
        
        for field in fields_to_smooth:
            values = []
            valid_indices = []
            
            for idx, frame in enumerate(self.frame_data):
                if frame[field] is not None:
                    values.append(frame[field])
                    valid_indices.append(idx)
            
            if len(values) < window_size:
                continue
            
            # 应用平滑
            values_array = np.array(values)
            
            if method == 'savgol' and len(values) >= window_size:
                if window_size % 2 == 0:
                    window_size += 1
                smoothed = savgol_filter(values_array, window_size, 3)
            elif method == 'gaussian':
                smoothed = gaussian_filter1d(values_array, sigma=2)
            elif method == 'median':
                smoothed = medfilt(values_array, kernel_size=window_size)
            else:
                smoothed = values_array
            
            # 更新平滑后的值
            for i, idx in enumerate(valid_indices):
                self.frame_data[idx][field] = float(smoothed[i])
    
    def detect_gait_cycles(self, fps: float) -> List[Dict]:
        """
        检测步态周期
        
        Args:
            fps: 视频帧率
            
        Returns:
            步态周期列表
        """
        if len(self.frame_data) < 10:
            return []
        
        # 提取脚部垂直位置
        left_foot_y = []
        right_foot_y = []
        
        for frame in self.frame_data:
            if frame['left_foot_position'] is not None:
                left_foot_y.append(frame['left_foot_position'][1])
            else:
                left_foot_y.append(None)
            
            if frame['right_foot_position'] is not None:
                right_foot_y.append(frame['right_foot_position'][1])
            else:
                right_foot_y.append(None)
        
        # 检测脚触地（局部最大值，因为y轴向下）
        cycles = []
        
        # 左脚
        left_strikes = self._detect_foot_strikes(left_foot_y)
        # 右脚
        right_strikes = self._detect_foot_strikes(right_foot_y)
        
        # 合并并排序
        all_strikes = [(idx, 'left') for idx in left_strikes] + \
                     [(idx, 'right') for idx in right_strikes]
        all_strikes.sort()
        
        # 计算步态周期
        for i in range(len(all_strikes) - 1):
            cycle = {
                'start_frame': all_strikes[i][0],
                'end_frame': all_strikes[i + 1][0],
                'foot': all_strikes[i][1],
                'duration_frames': all_strikes[i + 1][0] - all_strikes[i][0],
                'duration_seconds': (all_strikes[i + 1][0] - all_strikes[i][0]) / fps
            }
            cycles.append(cycle)
        
        return cycles
    
    def _detect_foot_strikes(self, foot_y: List[Optional[float]]) -> List[int]:
        """检测脚触地时刻"""
        strikes = []
        window = 5
        
        for i in range(window, len(foot_y) - window):
            if foot_y[i] is None:
                continue
            
            # 检查是否为局部最大值（y坐标最大，即最下方）
            is_local_max = True
            for j in range(i - window, i + window + 1):
                if j == i or foot_y[j] is None:
                    continue
                if foot_y[j] > foot_y[i]:
                    is_local_max = False
                    break
            
            if is_local_max and (not strikes or i - strikes[-1] > 10):
                strikes.append(i)
        
        return strikes
    
    def calculate_cadence(self, fps: float) -> float:
        """
        计算步频 (步/分钟)
        
        Args:
            fps: 视频帧率
            
        Returns:
            步频
        """
        cycles = self.detect_gait_cycles(fps)
        
        if len(cycles) < 2:
            return 0.0
        
        # 计算平均步幅时间
        total_duration = sum(c['duration_seconds'] for c in cycles)
        avg_stride_duration = total_duration / len(cycles)
        
        if avg_stride_duration == 0:
            return 0.0
        
        # 步频 = 60 / 平均步幅时间
        cadence = 60.0 / avg_stride_duration
        
        return cadence
    
    def get_summary_statistics(self) -> Dict:
        """
        获取总体统计数据
        
        Returns:
            统计摘要
        """
        if not self.frame_data:
            return {}
        
        summary = {}
        
        # 统计字段
        fields = [
            'trunk_tilt', 'left_knee_flexion', 'right_knee_flexion',
            'left_ankle_angle', 'right_ankle_angle',
            'left_hip_adduction', 'right_hip_adduction',
            'pelvic_drop', 'left_calcaneal_angle', 'right_calcaneal_angle'
        ]
        
        for field in fields:
            values = [f[field] for f in self.frame_data if f[field] is not None]
            
            if values:
                summary[field] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'samples': len(values)
                }
            else:
                summary[field] = None
        
        return summary
    
    def evaluate_parameter(self, param_name: str, value: float) -> str:
        """
        评估单个参数
        
        Args:
            param_name: 参数名称
            value: 参数值
            
        Returns:
            评级: 'good', 'mediocre', 'bad'
        """
        if value is None:
            return 'unknown'
        
        # 根据参数类型评估
        if param_name in ['trunk_tilt', 'left_knee_flexion', 'right_knee_flexion']:
            return self._evaluate_with_ranges(value, param_name)
        elif param_name in ['left_ankle_angle', 'right_ankle_angle']:
            return self._evaluate_ankle(value)
        elif param_name in ['left_hip_adduction', 'right_hip_adduction']:
            return self._evaluate_hip(value)
        elif param_name == 'pelvic_drop':
            return self._evaluate_pelvic(value)
        elif param_name in ['left_calcaneal_angle', 'right_calcaneal_angle']:
            return self._evaluate_calcaneal(value)
        
        return 'unknown'
    
    def _evaluate_with_ranges(self, value: float, param_type: str) -> str:
        """使用范围评估"""
        if param_type == 'trunk_tilt':
            thresh = self.thresholds['trunk_tilt']
            if value <= thresh['good']:
                return 'good'
            elif thresh['mediocre_range'][0] <= value <= thresh['mediocre_range'][1]:
                return 'mediocre'
            else:
                return 'bad'
        
        elif 'knee_flexion' in param_type:
            thresh = self.thresholds['knee_flexion']
            if thresh['good_range'][0] <= value <= thresh['good_range'][1]:
                return 'good'
            elif thresh['mediocre_range'][0] <= value <= thresh['mediocre_range'][1]:
                return 'mediocre'
            else:
                return 'bad'
        
        return 'unknown'
    
    def _evaluate_ankle(self, value: float) -> str:
        """评估踝关节角度"""
        thresh = self.thresholds['ankle_plantar']
        if value <= thresh['good_max']:
            return 'good'
        elif thresh['mediocre_range'][0] <= value <= thresh['mediocre_range'][1]:
            return 'mediocre'
        else:
            return 'bad'
    
    def _evaluate_hip(self, value: float) -> str:
        """评估髋关节内收"""
        thresh = self.thresholds['hip_adduction']
        if value <= thresh['good_max']:
            return 'good'
        elif thresh['mediocre_range'][0] <= value <= thresh['mediocre_range'][1]:
            return 'mediocre'
        else:
            return 'bad'
    
    def _evaluate_pelvic(self, value: float) -> str:
        """评估骨盆下降"""
        thresh = self.thresholds['pelvic_drop']
        if value <= thresh['good_max']:
            return 'good'
        elif thresh['mediocre_range'][0] <= value <= thresh['mediocre_range'][1]:
            return 'mediocre'
        else:
            return 'bad'
    
    def _evaluate_calcaneal(self, value: float) -> str:
        """评估小腿内外翻"""
        thresh = self.thresholds['calcaneal_rotation']
        if value >= thresh['good_min']:
            return 'good'
        elif thresh['mediocre_range'][0] <= value <= thresh['mediocre_range'][1]:
            return 'mediocre'
        else:
            return 'bad'
    
    def evaluate_cadence(self, cadence: float) -> str:
        """评估步频"""
        thresh = self.thresholds['cadence']
        if thresh['good_range'][0] <= cadence <= thresh['good_range'][1]:
            return 'good'
        elif thresh['mediocre_range'][0] <= cadence <= thresh['good_range'][1]:
            return 'mediocre'
        else:
            return 'bad'

