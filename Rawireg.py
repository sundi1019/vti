import numpy as np
import math
from scipy import integrate

# 导入依赖的类
from SW import PiersonMoskowitzSpectrum
from DA import WaveDirectionalSpreadFunction
from RAW import WaveAddedResistanceCalculator


class WaveResistanceIntegral:
    """
    波浪附加阻力积分计算类
    
    实现公式：ω² ≈ 2∫₀^∞∫₀^2π S(ω)D(α)R_AW(ω,α)/ζ_a² dα dω
    
    该类整合了波浪谱、方向分布函数和波浪附加阻力计算，
    通过数值积分方法计算总的波浪附加阻力。
    """
    
    def __init__(self, params=None):
        """
        初始化波浪附加阻力积分计算类
        
        参数:
        params (dict, optional): 包含所有必要参数的字典，分为以下几部分：
            wave_spectrum_params: 波浪谱参数
                - Hs: 有义波高(m)
                - Tp: 峰值周期(s)
            directional_spread_params: 方向分布函数参数
                - alpha_p: 主相对波浪方向(弧度)
                - n: 扩展参数
            ship_params: 船舶参数
                - L_PP: 垂线间长(m)
                - B: 船宽(m)
                - C_B: 方形系数
                - T_f: 前垂线吃水(m)
                - T_a: 后垂线吃水(m)
                - k_yy: 纵摇回转半径与L_PP的比值
                - E_1: 水线入流角(弧度)
                - E_2: 水线出流角(弧度)
                - U: 船舶航速(m/s)
                - zeta_A: 波幅(m)
                - lambda_wave: 波长(m)
            integration_params: 积分参数
                - omega_range: 频率积分范围(起始, 结束, 点数)
                - alpha_range: 角度积分范围(起始, 结束, 点数)
                
        Raises:
            ValueError: 当参数无效时抛出异常
        """
        # 初始化各组件
        self.wave_spectrum = None
        self.directional_spread = None
        self.resistance_calculator = None
        
        # 积分参数
        self.omega_range = (0.1, 2.0)  # 默认频率范围
        self.alpha_range = (0.0, 2 * math.pi)  # 默认角度范围(0到2π)
        
        # 结果存储
        self.results = {}
        
        # 如果提供了参数，则设置参数（set_parameters会自动验证）
        if params is not None:
            self.set_parameters(params)
    
    def set_parameters(self, params):
        """
        设置计算参数并自动验证
        
        参数:
        params (dict): 包含所有必要参数的字典
        
        Raises:
            ValueError: 当参数无效时抛出异常
        """
        # 设置波浪谱参数
        if 'wave_spectrum_params' in params:
            self.wave_spectrum = PiersonMoskowitzSpectrum(**params['wave_spectrum_params'])
        
        # 设置方向分布函数参数
        if 'directional_spread_params' in params:
            self.directional_spread = WaveDirectionalSpreadFunction(**params['directional_spread_params'])
        
        # 设置船舶参数
        if 'ship_params' in params:
            self.resistance_calculator = WaveAddedResistanceCalculator(params['ship_params'])
        
        # 设置积分参数
        if 'integration_params' in params:
            if 'omega_range' in params['integration_params']:
                self.omega_range = params['integration_params']['omega_range']
            if 'alpha_range' in params['integration_params']:
                self.alpha_range = params['integration_params']['alpha_range']
        
        # 清空之前的计算结果
        self.results = {}
    
    
    def _integrand(self, omega, alpha):
        """
        计算积分核函数 S(ω)D(α)R_AW(ω,α)/ζ_a²
        
        参数:
        omega: 波浪频率(rad/s)
        alpha: 波浪方向(弧度)
        
        返回:
        float: 积分核函数值
        """
        # 计算波浪谱值
        S_omega = self.wave_spectrum.calculate(omega)
        
        # 计算方向分布函数值
        D_alpha = self.directional_spread.calculate(alpha)
        
        # 计算波浪附加阻力值
        R_AW = self.resistance_calculator.calculate(omega, alpha)
        
        # 获取波幅值
        zeta_A = self.resistance_calculator.zeta_A
        
        # 计算积分核函数值
        if zeta_A != 0:  # 避免除零错误
            integrand_value = S_omega * D_alpha * R_AW / (zeta_A ** 2)
        else:
            integrand_value = 0.0
        
        return integrand_value
    
    def calculate(self):
        """
        计算波浪附加阻力的双重积分
        
        返回:
        float: 总的波浪附加阻力积分结果
        
        Raises:
            ValueError: 当参数无效时抛出（通过set_parameters验证）
        """
        
        # 使用scipy.integrate.dblquad直接进行二重积分
        # dblquad(func, a, b, gfun, hfun, ...) 其中a,b是外层积分范围，gfun,hfun是内层积分范围
        integral_result, _ = integrate.dblquad(
            # 定义二维被积函数，注意dblquad要求函数形式为func(y, x)，即先alpha再omega
            lambda alpha, omega: self._integrand(omega, alpha),
            # 外层积分(omega)范围
            self.omega_range[0], self.omega_range[1],
            # 内层积分(alpha)范围 - 常数范围，所以用lambda函数返回固定值
            lambda _: 0,  # 下限函数
            lambda _: 2 * math.pi,  # 上限函数
            # 精度控制参数
            epsabs=1e-6, epsrel=1e-6
        )
        
        # 根据公式乘以系数2
        total_result = 2 * integral_result
        
        # 保存计算结果
        self.results = {
            'total_wave_resistance': total_result,
            'integration_method': 'dblquad',
            'omega_range': self.omega_range,
            'alpha_range': self.alpha_range,
            'integration_time': None,  # 可以在这里添加积分时间计时
            'wave_spectrum_params': {
                'Hs': self.wave_spectrum.Hs,
                'Tp': self.wave_spectrum.Tp
            },
            'directional_spread_params': {
                'alpha_p': self.directional_spread.alpha_p,
                'n': self.directional_spread.n
            }
        }
        
        return total_result
    
    def get_results(self):
        """
        获取计算结果
        
        返回:
        dict: 包含计算结果的字典
        
        异常:
        ValueError: 当尚未进行计算时抛出
        """
        if not self.results:
            raise ValueError("尚未进行计算，请先调用calculate方法")
        
        return self.results.copy()


if __name__ == "__main__":
    # 示例1：完整参数初始化并计算
    params = {
        'wave_spectrum_params': {
            'Hs': 3.0,  # 有义波高(m)
            'Tp': 8.0  # 峰值周期(s)
        },
        'directional_spread_params': {
            'alpha_p': 0.0,  # 主方向(0表示顺浪)
            'n': 2  # 风浪扩展参数
        },
        'ship_params': {
            'L_PP': 175,  # 垂线间长(m)
            'B': 25,  # 船宽(m)
            'C_B': 0.75,  # 方形系数
            'T_f': 8.5,  # 前垂线吃水(m)
            'T_a': 9.0,  # 后垂线吃水(m)
            'k_yy': 0.25,  # 纵摇回转半径比
            'E_1': 0.52,  # 入流角(弧度)
            'E_2': 0.35,  # 出流角(弧度)
            'U': 10,  # 船速(m/s)
            'zeta_A': 2,  # 波幅(m)
            'lambda_wave': 150  # 波长(m)
        },
        'integration_params': {
            'omega_range': (0.0, math.inf, 100),  # 频率范围(起始,结束,点数)
            'alpha_range': (0.0, 2 * math.pi, 60)  # 角度范围(0到2π,点数)
        }
    }
    
    # 创建实例并计算（验证在set_parameters中自动执行）
    try:
        integral_calculator = WaveResistanceIntegral(params)
        print("示例1 - 参数验证:")
        print("验证结果: 所有参数验证通过")
    except ValueError as e:
        print(f"验证结果: {e}")
    
    # 使用不同的积分方法进行计算
    print("\n示例2 - 使用dblquad方法计算:")
    result = integral_calculator.calculate()
    print(f"波浪附加阻力积分结果: {result:.4f}")
    
    # 示例4：获取详细结果
    print("\n示例4 - 详细计算结果:")
    results = integral_calculator.get_results()
    print(f"积分方法: {results['integration_method']}")
    print(f"波浪谱参数 - Hs: {results['wave_spectrum_params']['Hs']}m, Tp: {results['wave_spectrum_params']['Tp']}s")
    print(f"方向分布参数 - alpha_p: {results['directional_spread_params']['alpha_p']}rad, n: {results['directional_spread_params']['n']}")
    
    # 示例5：修改部分参数后重新计算
    print("\n示例5 - 修改参数后重新计算:")
    new_params = {
        'wave_spectrum_params': {
            'Hs': 4.0,  # 增加有义波高
            'Tp': 10.0  # 增加峰值周期
        },
        'directional_spread_params': {
            'alpha_p': math.pi,  # 改为顶浪
            'n': 3  # 涌浪扩展参数
        }
    }
    try:
        integral_calculator.set_parameters(new_params)
        result_modified = integral_calculator.calculate()
        print(f"修改参数后的波浪附加阻力积分结果: {result_modified:.4f}")
    except ValueError as e:
        print(f"参数修改后验证失败: {e}")