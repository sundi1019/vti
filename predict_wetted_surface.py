#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
船舶湿表面积预测模块

功能：
- 基于B-Spline模型预测船舶湿表面积
- 支持通过参数指定模型类型和排水量
- 不使用交互式输入，适合集成到其他系统
"""

import joblib
from scipy.interpolate import splev
import os


class ShipWettedSurfacePredictor:
    """
    船舶湿表面积预测器
    
    用于基于B-Spline模型预测船舶湿表面积
    支持带PSV和不带PSV两种模型选择
    """
    
    def __init__(self):
        """
        初始化船舶湿表面积预测器
        设置默认模型类型和参数
        """
        self.model_type = None  # 模型类型
        self.disp_m3 = None     # 排水量(m³)
        
        # 模型文件配置
        self.model_file = 'ship_wetted_surface_models.joblib'
    
    def set_parameters(self, model_type, disp_m3):
        """
        设置计算参数
        
        参数:
            model_type (str): 模型类型，必须是'with_PSV'或'without_PSV'
            disp_m3 (float): 排水量(m³)，必须大于0
        
        异常:
            ValueError: 当参数值不在有效范围内时抛出
        """
        # 验证模型类型
        if model_type not in ['with_PSV', 'without_PSV']:
            raise ValueError(f"无效的模型类型: {model_type}。必须是 'with_PSV' 或 'without_PSV'")
        self.model_type = model_type
        
        # 验证排水量
        if not isinstance(disp_m3, (int, float)) or disp_m3 <= 0:
            raise ValueError(f"排水量必须大于0: {disp_m3}")
        self.disp_m3 = disp_m3
    
    def calculate(self):
        """
        执行船舶湿表面积预测计算
        
        返回:
            dict: 包含预测结果的字典
            {
                'wetted_surface': 预测的湿表面积 (m²),
                'model_type': 使用的模型类型,
                'disp_m3': 输入的排水量 (m³)
            }
        
        异常:
            ValueError: 当必要参数未设置时抛出
            FileNotFoundError: 当模型文件不存在时抛出
            Exception: 当预测过程中发生其他错误时抛出
        """
        # 验证参数是否已设置
        if self.model_type is None or self.disp_m3 is None:
            raise ValueError("所有必要参数未设置，请先调用set_parameters方法")
        
        # 构建完整的模型文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_file_path = os.path.join(script_dir, self.model_file)
        
        # 检查模型文件是否存在
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"找不到模型文件: {model_file_path}")
        
        # 加载模型
        try:
            models = joblib.load(model_file_path)
            
            # 检查模型是否完整
            if 'with_PSV' not in models or 'without_PSV' not in models:
                raise ValueError("模型文件不完整，缺少必要的模型数据")
            
            # 获取选定的模型
            tck_model = models[self.model_type]
            
            # 执行预测
            wetted_surface = float(splev(self.disp_m3, tck_model))
            
            # 返回结果
            return {
                'wetted_surface': wetted_surface,
                'model_type': self.model_type,
                'disp_m3': self.disp_m3
            }
            
        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise Exception(f"预测过程中发生错误: {str(e)}")


def predict_wetted_surface(model_type, disp_m3):
    """
    便捷函数：直接预测船舶湿表面积
    
    参数:
        model_type (str): 模型类型，必须是'with_PSV'或'without_PSV'
        disp_m3 (float): 排水量(m³)，必须大于0
    
    返回:
        dict: 包含预测结果的字典
    
    异常:
        当发生错误时抛出相应的异常
    """
    predictor = ShipWettedSurfacePredictor()
    predictor.set_parameters(model_type, disp_m3)
    return predictor.calculate()


# 示例用法
def example():
    """
    示例：展示如何使用ShipWettedSurfacePredictor类进行船舶湿表面积预测
    """
    # 创建预测器实例
    predictor = ShipWettedSurfacePredictor()
    
    # 设置参数并计算 - 测试用例1：使用带PSV的模型
    print("===== 测试用例1：带PSV模型 =====")
    predictor.set_parameters(model_type='with_PSV', disp_m3=10000.0)
    result1 = predictor.calculate()
    print(f"模型类型: {result1['model_type']}")
    print(f"排水量: {result1['disp_m3']:.0f} m³")
    print(f"预测湿表面积: {result1['wetted_surface']:.2f} m²")
    print()
    
    # 使用便捷方法 - 测试用例2：使用不带PSV的模型
    print("===== 测试用例2：不带PSV模型 =====")
    result2 = predict_wetted_surface(model_type='without_PSV', disp_m3=8500.0)
    print(f"模型类型: {result2['model_type']}")
    print(f"排水量: {result2['disp_m3']:.0f} m³")
    print(f"预测湿表面积: {result2['wetted_surface']:.2f} m²")
    print()
    
    # 测试用例3：不同排水量
    print("===== 测试用例3：不同排水量 =====")
    result3 = predict_wetted_surface(model_type='without_PSV', disp_m3=15000.0)
    print(f"模型类型: {result3['model_type']}")
    print(f"排水量: {result3['disp_m3']:.0f} m³")
    print(f"预测湿表面积: {result3['wetted_surface']:.2f} m²")


if __name__ == "__main__":
    example()