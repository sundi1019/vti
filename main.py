#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主程序入口

该模块负责从配置文件读取参数，调用Radd和Pds模块进行计算，
并输出计算结果。

注意事项：
- 运行前请确保已安装PyYAML库：pip install pyyaml
- 配置文件中的参数单位需严格按照注释说明填写
"""

import os
import sys
import yaml
import numpy as np
import math

# 添加当前目录到Python路径，确保能正确导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Radd import RaddCalculator, calculate_radd
from Pds import PdsCalculator
from predict_power import predict_ship_performance as predict_power_ship_performance
from predict_wetted_surface import predict_wetted_surface


def load_config(config_file='config.yaml'):
    """
    从配置文件中加载参数
    
    Args:
        config_file (str): 配置文件路径
        
    Returns:
        dict: 包含所有配置参数的字典
    """
    # 确保配置文件存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    # 读取YAML配置文件
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config


def predict_ship_performance(config):
    """
    预测船舶性能，计算交付功率P_DT和总效率eta_T
    
    Args:
        config (dict): 配置参数字典
        
    Returns:
        dict: 包含预测结果的字典
        {"power_DT": float, "eta_T": float}，如果预测失败返回None
    """
    # 从配置中获取model_type，如果没有则默认为'with_PSV'
    model_type = config.get('model_type', 'with_PSV')
    # 使用配置中的U作为航速(m/s)，Delta作为排水量
    U = config.get('U', 0.0)
    Delta = config.get('Delta', 0.0)
    
    # 调用预测函数获取船舶性能数据
    try:
        print(f"正在调用船舶性能预测，模型类型: {model_type}，航速: {U} m/s，排水量: {Delta} m³")
        performance_result = predict_power_ship_performance(model_type=model_type, speed_ms=U, disp_m3=Delta)
        power_DT = performance_result['power_DT']
        eta_T = performance_result['eta_T']
        print(f"船舶性能预测结果：交付功率P_DT = {power_DT:.2f} kW，总效率eta_T = {eta_T:.4f}")
        return {"power_DT": power_DT, "eta_T": eta_T}
    except Exception as e:
        print(f"船舶性能预测出错: {e}")
        return None


def predict_ship_wetted_surface(config):
    """
    预测船舶湿表面积
    
    Args:
        config (dict): 配置参数字典
        
    Returns:
        float: 湿表面积(m²)，如果预测失败返回None
    """
    # 从配置中获取model_type，如果没有则默认为'with_PSV'
    model_type = config.get('model_type', 'with_PSV')
    # 使用配置中的Delta作为排水量(m³)
    Delta = config.get('Delta', 0.0)
    
    # 检查必要参数
    if Delta <= 0:
        print("错误：排水量(Delta)必须大于0")
        return None
    
    # 调用计算函数获取湿表面积
    try:
        print(f"正在计算船舶湿表面积，模型类型: {model_type}，排水量: {Delta} m³")
        result_dict = predict_wetted_surface(model_type=model_type, disp_m3=Delta)
        wet_surface_area = result_dict['wetted_surface']
        print(f"湿表面积计算结果: {wet_surface_area:.2f} m²")
        return wet_surface_area
    except Exception as e:
        print(f"湿表面积计算出错: {e}")
        return None


def prepare_radd_params(config):
    """
    准备Radd计算所需的参数
    
    Args:
        config (dict): 配置参数字典
        
    Returns:
        dict: 格式化后的Radd参数字典
    """
    
    # 创建Radd计算所需的参数字典
    radd_params = {}
    
    # 构建波浪谱参数
    radd_params['wave_spectrum_params'] = {
        'Hs': config.get('Hs', 2.0),
        'Tp': config.get('Tp', 10.0)
    }
    
    # 构建方向扩散参数
    radd_params['directional_spread_params'] = {
        'alpha_p': config.get('alpha_p', 0.7854),
        'n': config.get('n', 3.0)
    }
    
    # 构建船舶主参数
    radd_params['ship_params'] = {
        'L_PP': config['L_PP'],
        'B': config['B'],
        'C_B': config['C_B'],
        'T_f': config['T_f'],
        'T_a': config['T_a'],
        'k_yy': config['k_yy'],
        'E_1': config['E_1'],
        'E_2': config['E_2'],
        'U': config.get('U', 5.0),
        'zeta_A': config.get('zeta_A', 1.0),
        'lambda_wave': config.get('lambda_wave', 100.0)
    }
    
    # 构建积分参数
    radd_params['integration_params'] = {
        'omega_range': (config.get('omega_min', 0.0), config.get('omega_max', math.inf)),
        'alpha_range': (config.get('alpha_min', 0.0), config.get('alpha_max', 2 * math.pi))
    }
    
    # 添加风阻计算参数（扁平结构）
    wind_params = ['L_OA', 'A_OD', 'A_XV', 'A_LV', 'C_MC', 
                  'H_BR', 'H_C', 'mu', 'psi_rel', 'V_rel', 'V_G']
    for param in wind_params:
        if param in config:
            radd_params[param] = config[param]
    
    # 添加温度影响参数（扁平结构）
    # 注意：nt参数已经通过predict_power计算设置，不需要从配置文件中重复读取
    temp_params = ['velocity', 'Lpp', 'L_wl', 'wet_surface_area', 
                  'temperature', 'kS', 'Pds', 'T', 'Delta', 
                  'A_m', 'aft_body_shape']
    for param in temp_params:
        if param in config:
            radd_params[param] = config[param]
    
    # 确保B参数也在顶层（风阻和温度计算可能需要）
    radd_params['B'] = config['B']
    
    return radd_params


def calculate_pds(config):
    """
    计算传递功率Pds
    
    Args:
        config (dict): 配置参数字典
        
    Returns:
        float: 传递功率值(kW)
    """
    # 创建Pds计算器实例
    pds_calculator = PdsCalculator()
    
    # 获取Pds计算所需参数
    pm = config.get('pm', 0.0)
    eta_trans = config.get('eta_trans', None)
    
    # 计算Pds
    pds = pds_calculator.calculate_pds(pm=pm, eta_trans=eta_trans)
    
    return pds

def main():
    """
    主函数
    """
    print("船舶天气附加阻力及传递功率计算程序")
    print("===================================")
    
    try:
        # 加载配置文件
        config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        print(f"正在读取配置文件: {config_file_path}")
        config = load_config(config_file_path)
        
        print(f"成功读取配置文件: {config_file_path}")
        
        # 打印船舶基本参数信息
        print("\n===== 船舶基本参数 =====")
        print(f"船长 (L_PP): {config.get('L_PP', 'N/A')} m")
        print(f"船宽 (B): {config.get('B', 'N/A')} m")
        print(f"方形系数 (C_B): {config.get('C_B', 'N/A')}")
        print(f"首吃水 (T_f): {config.get('T_f', 'N/A')} m")
        print(f"尾吃水 (T_a): {config.get('T_a', 'N/A')} m")
        print(f"全船长 (L_OA): {config.get('L_OA', 'N/A')} m")
        print(f"水线长 (L_wl): {config.get('L_wl', 'N/A')} m")
        print(f"平均吃水 (T): {config.get('T', 'N/A')} m")
        print(f"船舶排水量 (Delta): {config.get('Delta', 'N/A')} m³")
        print(f"湿表面积系数 (kS): {config.get('kS', 'N/A')}")
        print(f"船体中剖面积 (A_m): {config.get('A_m', 'N/A')} m²")
        print(f"船尾形状 (aft_body_shape): {config.get('aft_body_shape', 'N/A')}")
        
        # 计算船舶湿表面积
        # 使用封装好的函数计算湿表面积
        wet_surface_area = predict_ship_wetted_surface(config)
        
        # 计算船舶性能（交付功率P_DT和总效率eta_T）
        performance_result = predict_ship_performance(config)
        
        # 计算传递功率Pds
        pds_value = calculate_pds(config)
        print(f"\n传递功率 (Pds): {pds_value:.2f} kW")
        
        # 准备Radd计算参数
        print("\n准备计算天气附加阻力...")
        radd_params = prepare_radd_params(config)
        
        # 更新Radd参数中的船舶湿表面积
        radd_params['wet_surface_area'] = wet_surface_area
        # 更新Radd参数中的Pds值
        radd_params['Pds'] = pds_value
        
        radd_params['nt'] = performance_result['eta_T']
        
        # 计算Radd（单位：N）
        radd_value = calculate_radd(radd_params)
        # 显示为kN以保持可读性
        print(f"\n总附加阻力 (Radd): {radd_value:.2f} N")
        
        # 计算VTI公式
        Vw = config.get('U', 5.0)  # 船速Vw (m/s)
        Delta = config.get('Delta', 0.0)  # 排水量m³
        
        # 计算Penv = Radd * Vw / nt，并转换为kW
        Penv = (radd_value * Vw / performance_result['eta_T']) / 1000  # 转换为kW
        
        # 计算VTI，需要将Pds也转换为W进行计算
        VTI = (pds_value  - Penv) /  performance_result['power_DT']
        
        print(f"\n===== VTI计算结果 =====")
        print(f"Pds: {pds_value:.2f} kW")
        print(f"Penv: {Penv:.2f} kW")
        print(f"P0 (P_DT): {performance_result['power_DT']:.2f} kW")
        print(f"Vw: {Vw:.2f} m/s")
        print(f"Δ: {Delta} m³")
        print(f"VTI = (Pds - Penv) / power_DT = {VTI:.6f}")
        
        print("\n计算完成！")
        
    except ImportError as e:
        print(f"导入模块时出错: {e}")
        print("请确保已安装PyYAML库: pip install pyyaml")
    except yaml.YAMLError as e:
        print(f"解析YAML配置文件时出错: {e}")
    except FileNotFoundError as e:
        print(f"配置文件未找到: {e}")
    except KeyError as e:
        print(f"配置文件缺少必要的键: {e}")
    except Exception as e:
        print(f"程序运行时出错: {e}")
        import traceback
        print("错误详情:")
        traceback.print_exc()


if __name__ == "__main__":
    main()