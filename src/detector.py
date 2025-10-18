"""
姿态检测模块 - 使用YOLO进行姿态估计
Pose Detector - Using YOLO for pose estimation
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
import torch


class PoseDetector:
    """YOLO姿态检测器"""
    
    def __init__(self, model_name: str = "yolov8x-pose.pt", 
                 confidence: float = 0.5, 
                 device: str = "cuda"):
        """
        初始化姿态检测器
        
        Args:
            model_name: YOLO模型名称
            confidence: 检测置信度阈值
            device: 运行设备 (cuda 或 cpu)
        """
        self.device = device
        self.confidence = confidence
        
        # 检查设备可用性
        if device == "cuda" and not torch.cuda.is_available():
            print("警告: CUDA不可用，切换到CPU模式")
            self.device = "cpu"
        
        # 加载YOLO模型
        print(f"加载模型: {model_name} ...")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        print(f"模型已加载到 {self.device}")
        
        # COCO关键点索引
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        检测单帧图像中的人体姿态
        
        Args:
            frame: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个结果包含关键点和置信度
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        detections = []
        
        for result in results:
            if result.keypoints is None:
                continue
                
            keypoints = result.keypoints.data.cpu().numpy()
            
            for person_kpts in keypoints:
                if len(person_kpts) != 17:
                    continue
                
                # 提取关键点信息
                person_data = {
                    'keypoints': {},
                    'bbox': None,
                    'confidence': 0.0
                }
                
                total_conf = 0.0
                valid_kpts = 0
                
                for idx, (x, y, conf) in enumerate(person_kpts):
                    kpt_name = self.keypoint_names[idx]
                    person_data['keypoints'][kpt_name] = {
                        'x': float(x),
                        'y': float(y),
                        'confidence': float(conf)
                    }
                    
                    if conf > 0.3:  # 有效关键点
                        total_conf += conf
                        valid_kpts += 1
                
                # 计算平均置信度
                if valid_kpts > 0:
                    person_data['confidence'] = total_conf / valid_kpts
                
                # 获取边界框
                if result.boxes is not None and len(result.boxes) > 0:
                    box = result.boxes[0].xyxy[0].cpu().numpy()
                    person_data['bbox'] = box.tolist()
                
                detections.append(person_data)
        
        return detections
    
    def draw_pose(self, frame: np.ndarray, detections: List[Dict], 
                  show_confidence: bool = True) -> np.ndarray:
        """
        在图像上绘制姿态骨架
        
        Args:
            frame: 输入图像
            detections: 检测结果
            show_confidence: 是否显示置信度
            
        Returns:
            标注后的图像
        """
        annotated = frame.copy()
        
        # 定义骨架连接
        skeleton = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('nose', 'left_eye'),
            ('nose', 'right_eye'),
            ('left_eye', 'left_ear'),
            ('right_eye', 'right_ear'),
        ]
        
        for person in detections:
            keypoints = person['keypoints']
            
            # 绘制骨架连接
            for start, end in skeleton:
                if start in keypoints and end in keypoints:
                    start_kpt = keypoints[start]
                    end_kpt = keypoints[end]
                    
                    if start_kpt['confidence'] > 0.3 and end_kpt['confidence'] > 0.3:
                        pt1 = (int(start_kpt['x']), int(start_kpt['y']))
                        pt2 = (int(end_kpt['x']), int(end_kpt['y']))
                        cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)
            
            # 绘制关键点
            for kpt_name, kpt in keypoints.items():
                if kpt['confidence'] > 0.3:
                    x, y = int(kpt['x']), int(kpt['y'])
                    cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
                    
                    if show_confidence:
                        conf_text = f"{kpt['confidence']:.2f}"
                        cv2.putText(annotated, conf_text, (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        return annotated
    
    def get_keypoint_array(self, detection: Dict, keypoint_name: str) -> Optional[np.ndarray]:
        """
        获取关键点的numpy数组
        
        Args:
            detection: 检测结果
            keypoint_name: 关键点名称
            
        Returns:
            关键点坐标 [x, y] 或 None
        """
        if keypoint_name not in detection['keypoints']:
            return None
        
        kpt = detection['keypoints'][keypoint_name]
        
        if kpt['confidence'] < 0.3:
            return None
        
        return np.array([kpt['x'], kpt['y']])

