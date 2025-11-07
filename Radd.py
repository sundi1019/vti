"""
Radd - 天气引起的附加阻力计算模块

该模块实现了由天气引起的附加阻力计算，包括：
- 波浪附加阻力 (Rawireg)
- 风阻力 (Rwind)
- 温度变化引起的阻力 (Rtemp)

总附加阻力计算公式：Radd = Rawireg + Rwind + Rtemp
"""

import numpy as np
from Rawireg import WaveResistanceIntegral
from Rwind import WindResistanceCalculator
from Rtemp import FluidDynamicsCalculator


class RaddCalculator:
    """
    天气附加阻力计算器类
    
    该类集成了波浪附加阻力、风阻力和温度变化阻力的计算，
    提供统一的接口计算总的天气附加阻力。
    """
    
    def __init__(self):
        """
        初始化Radd计算器
        """
        # 初始化各个阻力计算器
        self.rawireg_calculator = WaveResistanceIntegral()
        self.rwind_calculator = WindResistanceCalculator()
        self.rtemp_calculator = FluidDynamicsCalculator()
    
    def set_parameters(self, ship_params):
        """
        设置计算所需的船舶和环境参数
        
        Args:
            ship_params (dict): 包含船舶参数和环境参数的字典
                               需包含所有子模块所需的参数
                               结构如下：
                               {
                                   # WaveResistanceIntegral所需参数
                                   'wave_spectrum_params': {
                                       'Hs': 有义波高(m),
                                       'Tp': 峰值周期(s)
                                   },
                                   'directional_spread_params': {
                                       'alpha_p': 主相对波浪方向(弧度),
                                       'n': 扩展参数
                                   },
                                   'ship_params': {
                                       'L_PP': 垂线间长(m),
                                       'B': 船宽(m),
                                       'C_B': 方形系数,
                                       'T_f': 前垂线吃水(m),
                                       'T_a': 后垂线吃水(m),
                                       'k_yy': 纵摇回转半径与L_PP的比值,
                                       'E_1': 水线入流角(弧度),
                                       'E_2': 水线出流角(弧度),
                                       'U': 船舶航速(m/s),
                                       'zeta_A': 波幅(m),
                                       'lambda_wave': 波长(m)
                                   },
                                   'integration_params': {
                                       'omega_range': 频率积分范围,
                                       'alpha_range': 角度积分范围
                                   },
                                   
                                   # WindResistanceCalculator所需参数
                                   'L_OA': 船舶总长(m),
                                   'A_OD': 上甲板以上上层建筑侧向投影面积(m²),
                                   'A_XV': 水线以上包括上层建筑的横向投影面积(m²),
                                   'A_LV': 水线以上包括上层建筑的侧向投影面积(m²),
                                   'C_MC': 船中到侧向投影面积A_LV重心的水平距离(m),
                                   'H_BR': 上层建筑顶部高度(m),
                                   'H_C': 从水线到侧向投影面积A_LV重心的高度(m),
                                   'mu': 平滑范围角度(度),
                                   'psi_rel': 相对风向角度(度),
                                   'V_rel': 相对风速(m/s),
                                   'V_G': 船舶对地速度(m/s),
                                   
                                   # FluidDynamicsCalculator所需参数
                                   'velocity': 对水速度(m/s)
                                   'Lpp': 垂线间长(m)
                                   'temperature': 水温(°C),
                                   'L_wl': 水线长(m),
                                   'kS': 船体表面粗糙度(m),
                                   'wet_surface_area': 船舶湿表面积(m²),
                                   'Pds': 实测交付功率(kW),
                                   'nt': 船舶总效率,
                                   'T': 型深(m),
                                   'Delta': 型排水量(m³),
                                   'A_m': 中剖面面积(m²),
                                   'aft_body_shape': 尾部形状
                               }
        """
        # 保存参数字典，以便在calculate方法中使用
        self.ship_params = ship_params
        
        # WaveResistanceIntegral需要特定结构的参数，直接传递整个ship_params
        self.rawireg_calculator.set_parameters(ship_params)
        
        # WindResistanceCalculator需要扁平结构的参数，直接传递整个ship_params
        self.rwind_calculator.set_parameters(ship_params)
        
        # FluidDynamicsCalculator需要扁平结构的参数，直接传递整个ship_params进行初始设置
        self.rtemp_calculator.set_parameters(ship_params)
    
    def calculate_rawireg(self):
        """
        计算波浪附加阻力
        
        Returns:
            float: 波浪附加阻力值
        """
        return self.rawireg_calculator.calculate()
    
    def calculate_rwind(self):
        """
        计算风阻力
        
        Returns:
            float: 风阻力值
        """
        return self.rwind_calculator.calculate()
    
    def calculate_rtemp(self):
        """
        计算温度变化引起的阻力
        
        Returns:
            float: 温度变化阻力值
        """
        return self.rtemp_calculator.calculate()
    
    def calculate(self):
        """
        计算总的天气附加阻力
        
        根据公式：Radd = Rawireg + Rwind + Rtemp
        
        Returns:
            float: 总的天气附加阻力值
        """
        # 先计算波浪附加阻力和风阻力
        rawireg = self.calculate_rawireg()
        rwind = self.calculate_rwind()
        
        # 直接更新rtemp_calculator的实例变量，避免重新调用set_parameters
        self.rtemp_calculator.Rawirge = rawireg
        self.rtemp_calculator.Rwind = rwind
        
        # 计算温度变化阻力
        rtemp = self.calculate_rtemp()

        print(f"波浪附加阻力 (Rawireg): {rawireg:.2f} N")
        print(f"风阻力 (Rwind): {rwind:.2f} N")
        print(f"温度变化阻力 (Rtemp): {rtemp:.2f} N")

        # 计算总附加阻力
        radd = rawireg + rwind + rtemp
        
        return radd


# 提供便捷的函数接口
def calculate_radd(ship_params):
    """
    便捷函数：计算总的天气附加阻力
    
    Args:
        ship_params (dict): 包含船舶参数和环境参数的字典
    
    Returns:
        float: 总的天气附加阻力值
    """
    calculator = RaddCalculator()
    calculator.set_parameters(ship_params)
    return calculator.calculate()


if __name__ == "__main__":
    """
    主测试函数：演示Radd模块的使用
    """
    # 示例参数 - 完整包含所有子计算器所需的参数
    example_params = {
        # WaveResistanceIntegral参数
        'wave_spectrum_params': {
            'Hs': 2.0,  # 有义波高(m) - 正值
            'Tp': 10.0  # 峰值周期(s) - 正值
        },
        'directional_spread_params': {
            'alpha_p': np.pi/4,  # 主相对波浪方向(弧度)
            'n': 3.0  # 扩展参数
        },
        'ship_params': {
            'L_PP': 100.0,  # 垂线间长(m) - 正值
            'B': 15.0,  # 船宽(m) - 正值
            'C_B': 0.7,  # 方形系数 - 0-1之间
            'T_f': 8.0,  # 前垂线吃水(m) - 正值
            'T_a': 10.0,  # 后垂线吃水(m) - 正值
            'k_yy': 0.25,  # 纵摇回转半径与L_PP的比值 - 正值
            'E_1': 0.1,  # 水线入流角(弧度)
            'E_2': -0.1,  # 水线出流角(弧度)
            'U': 5.0,  # 船舶航速(m/s) - 正值
            'zeta_A': 1.0,  # 波幅(m) - 正值
            'lambda_wave': 100.0  # 波长(m) - 正值
        },
        'integration_params': {
            'omega_range': (0.2, 2.0),  # 频率积分范围 - 正值范围
            'alpha_range': (0, 2*np.pi)  # 角度积分范围（0到2π）
        },
        
        # WindResistanceCalculator参数
        'L_OA': 105.0,  # 船舶总长(m) - 正值
        'A_OD': 200.0,  # 上甲板以上上层建筑侧向投影面积(m²) - 正值
        'A_XV': 150.0,  # 水线以上包括上层建筑的横向投影面积(m²) - 正值
        'A_LV': 300.0,  # 水线以上包括上层建筑的侧向投影面积(m²) - 正值
        'C_MC': 5.0,  # 船中到侧向投影面积A_LV重心的水平距离(m)
        'H_BR': 20.0,  # 上层建筑顶部高度(m) - 正值
        'H_C': 12.0,  # 从水线到侧向投影面积A_LV重心的高度(m) - 正值
        'mu': 30.0,  # 平滑范围角度(度) - 正值
        'psi_rel': 45.0,  # 相对风向角度(度)
        'V_rel': 10.0,  # 相对风速(m/s) - 正值
        'V_G': 5.0,  # 船舶对地速度(m/s) - 正值
        
        # FluidDynamicsCalculator参数 - 完整参数集（符合校验规则）
        'velocity': 6.17,  # 对水速度(m/s) - 大于0，符合校验规则
        'Lpp': 100.0,  # 垂线间长(m) - 大于0，符合校验规则
        'L_wl': 100.0,  # 水线长(m) - 大于0，符合校验规则
        'wet_surface_area': 1800.0,  # 船舶湿表面积(m²) - 大于0，符合校验规则
        'temperature': 20.0,  # 水温(°C) - 在0-40°C范围内，符合校验规则
        'kS': 150e-6,  # 船体表面粗糙度(m) - 大于0，符合校验规则
        'Pds': 5000.0,  # 实测交付功率(kW) - 大于0，符合校验规则
        'nt': 0.5,  # 船舶总效率 - 在0-1之间，符合校验规则
        'B': 15.0,  # 型宽(m) - 大于0，符合校验规则
        'T': 12.0,  # 型深(m) - 大于0，符合校验规则
        'Delta': 10000.0,  # 型排水量(m³) - 大于0，符合校验规则
        'A_m': 120.0,  # 中剖面面积(m²) - 大于0，符合校验规则
        'aft_body_shape': 'normal sections'  # 尾部形状，符合字符串要求
    }
    
    # 使用类方法计算
    print("===== 使用RaddCalculator类计算 =====")
    calculator = RaddCalculator()
    calculator.set_parameters(example_params)
    
    radd = calculator.calculate()
    
    print(f"总天气附加阻力 (Radd): {radd:.2f} N")