#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
船舶性能预测模块

功能：
- 基于B-Spline模型预测船舶交付功率和总效率
- 支持通过参数指定模型类型、航速和排水量
- 不使用交互式输入，适合集成到其他系统
"""

import joblib
from scipy.interpolate import bisplev
import os


class ShipPerformancePredictor:
    """
    船舶性能预测器
    
    用于基于B-Spline模型预测船舶交付功率和总效率
    支持带PSV和不带PSV两种模型选择
    """
    
    def __init__(self):
        """
        初始化船舶性能预测器
        设置默认模型类型和参数
        """
        self.model_type = None  # 模型类型
        self.speed_ms = None    # 航速(m/s)
        self.disp_m3 = None     # 排水量(m³)
        
        # 模型文件配置
        self.model_files = {
            'with_PSV': 'ship_with_PSV_bspline_models.joblib',
            'without_PSV': 'ship_without_PSV_bspline_models.joblib'
        }
    
    def set_parameters(self, model_type, speed_ms, disp_m3):
        """
        设置计算参数
        
        参数:
            model_type (str): 模型类型，必须是'with_PSV'或'without_PSV'
            speed_ms (float): 航速(m/s)，必须为非负数
            disp_m3 (float): 排水量(m³)，必须大于0
        
        异常:
            ValueError: 当参数值不在有效范围内时抛出
        """
        # 验证模型类型
        if model_type not in ['with_PSV', 'without_PSV']:
            raise ValueError(f"无效的模型类型: {model_type}。必须是 'with_PSV' 或 'without_PSV'")
        self.model_type = model_type
        
        # 验证航速
        if not isinstance(speed_ms, (int, float)) or speed_ms < 0:
            raise ValueError(f"航速必须是非负数: {speed_ms}")
        self.speed_ms = speed_ms
        
        # 验证排水量
        if not isinstance(disp_m3, (int, float)) or disp_m3 <= 0:
            raise ValueError(f"排水量必须大于0: {disp_m3}")
        self.disp_m3 = disp_m3
    
    def calculate(self):
        """
        执行船舶性能预测计算
        
        返回:
            dict: 包含预测结果的字典
            {
                'power_DT': 预测的交付功率 (kW),
                'eta_T': 预测的总效率,
                'model_type': 使用的模型类型,
                'speed_kn': 输入的航速 (kn),
                'disp_m3': 输入的排水量 (m³)
            }
        
        异常:
            ValueError: 当必要参数未设置时抛出
            FileNotFoundError: 当模型文件不存在时抛出
            Exception: 当预测过程中发生其他错误时抛出
        """
        # 验证参数是否已设置
        if self.model_type is None or self.speed_ms is None or self.disp_m3 is None:
            raise ValueError("所有必要参数未设置，请先调用set_parameters方法")
        
        # 构建完整的模型文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # model_dir = os.path.join(script_dir, '..', 'predict_power_efficiency')
        model_file_path = os.path.join(script_dir, self.model_files[self.model_type])
        
        # 检查模型文件是否存在
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"找不到模型文件: {model_file_path}")
        
        # 加载模型
        try:
            saved_data = joblib.load(model_file_path)
            models = saved_data['models']
        except Exception as e:
            raise Exception(f"加载模型时出错: {str(e)}")
        
        # 直接使用m/s单位
        speed_ms = self.speed_ms
        # 执行预测
        try:
            # 使用bisplev和加载的tck模型进行预测
            power = float(bisplev(speed_ms, self.disp_m3, models['power_DT']))
            eta_direct = float(bisplev(speed_ms, self.disp_m3, models['eta_DS']))
            
            # 构建返回结果
            result = {
                'power_DT': power,
                'eta_T': eta_direct,
                'model_type': self.model_type,
                'speed_ms': self.speed_ms,
                'disp_m3': self.disp_m3
            }
            
            return result
        except Exception as e:
            raise Exception(f"预测过程中出错: {str(e)}")
    
    def predict(self, model_type, speed_ms, disp_m3):
        """
        便捷方法：一次性设置参数并执行预测
        
        参数:
            model_type (str): 模型类型，必须是'with_PSV'或'without_PSV'
            speed_ms (float): 航速(m/s)，必须为非负数
            disp_m3 (float): 排水量(m³)，必须大于0
        
        返回:
            dict: 包含预测结果的字典
        """
        self.set_parameters(model_type, speed_ms, disp_m3)
        return self.calculate()


def predict_ship_performance(model_type='with_PSV', speed_ms=0.0, disp_m3=0.0):
    """
    兼容旧接口的函数，调用ShipPerformancePredictor类进行预测
    
    参数:
        model_type (str): 模型类型，可选 'with_PSV' 或 'without_PSV'
        speed_ms (float): 航速 (m/s)
        disp_m3 (float): 排水量 (m³)
    
    返回:
        dict: 包含预测结果的字典
    """
    predictor = ShipPerformancePredictor()
    return predictor.predict(model_type, speed_ms, disp_m3)


# 示例用法
def example():
    """
    示例：展示如何使用ShipPerformancePredictor类进行船舶性能预测
    """
    # 创建预测器实例
    predictor = ShipPerformancePredictor()
    
    # 设置参数并计算 - 测试用例1：使用带PSV的模型
    print("===== 测试用例1：带PSV模型 =====")
    # 15.0 kn ≈ 7.71666 m/s
    predictor.set_parameters(model_type='with_PSV', speed_ms=12.0, disp_m3=10000.0)
    result1 = predictor.calculate()
    print(f"模型类型: {result1['model_type']}")
    print(f"航速: {result1['speed_ms']} m/s")
    print(f"排水量: {result1['disp_m3']:.0f} m³")
    print(f"预测交付功率 P_DT: {result1['power_DT']:.2f} kW")
    print(f"预测总效率 eta_T: {result1['eta_T']:.4f}")
    print()
    
    # 使用便捷方法 - 测试用例2：使用不带PSV的模型
    print("===== 测试用例2：不带PSV模型 =====")
    # 12.0 kn ≈ 6.17333 m/s
    result2 = predictor.predict(model_type='without_PSV', speed_ms=6.17333, disp_m3=8500.0)
    print(f"模型类型: {result2['model_type']}")
    print(f"航速: {result2['speed_ms']} m/s")
    print(f"排水量: {result2['disp_m3']:.0f} m³")
    print(f"预测交付功率 P_DT: {result2['power_DT']:.2f} kW")
    print(f"预测总效率 eta_T: {result2['eta_T']:.4f}")


if __name__ == "__main__":
    example()