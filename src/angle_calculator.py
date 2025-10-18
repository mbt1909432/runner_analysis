"""
角度计算模块 - 计算关键点之间的角度
Angle Calculator - Calculate angles between keypoints
"""

import numpy as np
from typing import Tuple, Optional


class AngleCalculator:
    """计算人体关节角度"""
    
    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        计算三个点形成的角度
        
        Args:
            p1: 第一个点 [x, y]
            p2: 顶点（角的顶点） [x, y]
            p3: 第三个点 [x, y]
            
        Returns:
            角度（度）
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        # 计算向量的模
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        # 计算夹角
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    @staticmethod
    def calculate_vertical_angle(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        计算线段与垂直线的夹角
        
        Args:
            p1: 起点 [x, y]
            p2: 终点 [x, y]
            
        Returns:
            与垂直线的夹角（度）
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if dy == 0:
            return 90.0
        
        angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
        
        # 判断前倾还是后倾
        if dx > 0:  # 向右倾斜（前倾）
            return angle
        else:  # 向左倾斜（后倾）
            return -angle
    
    @staticmethod
    def calculate_trunk_tilt(shoulder: np.ndarray, hip: np.ndarray) -> float:
        """
        计算躯干倾斜角度
        
        Args:
            shoulder: 肩部位置 [x, y]
            hip: 髋部位置 [x, y]
            
        Returns:
            躯干倾斜角度（度），正值表示前倾
        """
        angle = AngleCalculator.calculate_vertical_angle(hip, shoulder)
        return abs(angle)
    
    @staticmethod
    def calculate_knee_flexion(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
        """
        计算膝关节屈曲角度
        
        Args:
            hip: 髋部位置 [x, y]
            knee: 膝盖位置 [x, y]
            ankle: 踝关节位置 [x, y]
            
        Returns:
            膝关节屈曲角度（度）
        """
        angle = AngleCalculator.calculate_angle(hip, knee, ankle)
        return 180 - angle  # 转换为屈曲角度
    
    @staticmethod
    def calculate_ankle_angle(knee: np.ndarray, ankle: np.ndarray, foot_point: np.ndarray) -> float:
        """
        计算踝关节角度
        
        Args:
            knee: 膝盖位置 [x, y]
            ankle: 踝关节位置 [x, y]
            foot_point: 脚部参考点 [x, y]
            
        Returns:
            踝关节角度（度）
        """
        angle = AngleCalculator.calculate_angle(knee, ankle, foot_point)
        return angle
    
    @staticmethod
    def calculate_hip_adduction(hip_left: np.ndarray, hip_right: np.ndarray, 
                                knee: np.ndarray, ankle: np.ndarray) -> float:
        """
        计算髋关节内收角度
        
        Args:
            hip_left: 左髋位置 [x, y]
            hip_right: 右髋位置 [x, y]
            knee: 膝盖位置 [x, y]
            ankle: 踝关节位置 [x, y]
            
        Returns:
            髋关节内收角度（度）
        """
        # 计算髋部中心
        hip_center = (hip_left + hip_right) / 2
        
        # 计算垂直参考线（从髋部中心向下）
        vertical_point = hip_center + np.array([0, 100])
        
        # 计算髋部到膝盖的线与垂直线的夹角
        v1 = vertical_point - hip_center
        v2 = knee - hip_center
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    @staticmethod
    def calculate_pelvic_drop(hip_left: np.ndarray, hip_right: np.ndarray) -> float:
        """
        计算骨盆下降角度
        
        Args:
            hip_left: 左髋位置 [x, y]
            hip_right: 右髋位置 [x, y]
            
        Returns:
            骨盆下降角度（度）
        """
        dx = hip_right[0] - hip_left[0]
        dy = hip_right[1] - hip_left[1]
        
        if dx == 0:
            return 0.0
        
        angle = np.degrees(np.arctan(abs(dy) / abs(dx)))
        return angle
    
    @staticmethod
    def calculate_calcaneal_angle(knee: np.ndarray, ankle: np.ndarray) -> float:
        """
        计算小腿内外翻角度（跟骨角度）
        
        Args:
            knee: 膝盖位置 [x, y]
            ankle: 踝关节位置 [x, y]
            
        Returns:
            小腿角度（度），正值表示外翻，负值表示内翻
        """
        angle = AngleCalculator.calculate_vertical_angle(ankle, knee)
        return angle
    
    @staticmethod
    def distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """
        计算两点之间的欧氏距离
        
        Args:
            p1: 第一个点 [x, y]
            p2: 第二个点 [x, y]
            
        Returns:
            距离
        """
        return np.linalg.norm(p1 - p2)
    
    @staticmethod
    def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        计算两点的中点
        
        Args:
            p1: 第一个点 [x, y]
            p2: 第二个点 [x, y]
            
        Returns:
            中点 [x, y]
        """
        return (p1 + p2) / 2

