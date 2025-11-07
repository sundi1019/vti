import numpy as np
import math

class PiersonMoskowitzSpectrum:
    """
    Pierson-Moskowitz波浪谱计算类
    
    该类实现了Pierson-Moskowitz波浪谱的计算，
    支持参数设置和谱计算功能。
    """
    
    def __init__(self, Hs=None, Tp=None):
        """
        初始化Pierson-Moskowitz波浪谱类
        
        参数:
        Hs: 有义波高(Significant wave height)，单位为m，默认为None
        Tp: 峰值周期(Peak period)，单位为s，默认为None
        """
        self.Hs = Hs
        self.Tp = Tp
        self.omega = None
        self.spectrum_value = None
        self.validate_parameters()
    
    def validate_parameters(self):
        """
        验证计算所需的参数是否已设置
        
        异常:
        ValueError: 当必要参数未设置时抛出
        """
        if self.Hs is None:
            raise ValueError("参数Hs(有义波高)未设置")
        if self.Tp is None:
            raise ValueError("参数Tp(峰值周期)未设置")
    

    
    def calculate(self, omega):
        """
        计算Pierson-Moskowitz波浪谱密度 (S(w))
        
        参数:
        omega: 单个频率值(ω)，单位为rad/s
        
        返回:
        S_eta_eta: 对应频率ω的波浪谱密度值，即公式中的S(ω)
        
        异常:
        ValueError: 当必要参数未设置时抛出
        TypeError: 当omega不是数值类型时抛出
        """
        
        # 确保omega是数值类型
        if not isinstance(omega, (int, float, np.number)):
            raise TypeError("参数omega必须是数值类型")
        
        # 计算峰值频率
        omega_p = 2 * math.pi / self.Tp
        
        # 实现PM谱公式: S_ηη(ω) = (5/16) * Hs² * ωp⁴ * ω⁻⁵ * exp(-(5/4)*(ω/ωp)⁻⁴)
        # 这对应于公式中的S(ω)
        term1 = (5/16) * self.Hs**2 * omega_p**4
        term2 = omega**(-5)
        exponent = -(5/4) * (omega / omega_p)**(-4)
        spectrum_value = term1 * term2 * math.exp(exponent)
        
        # 保存最后计算的值以便get_results使用
        self.omega = omega
        self.spectrum_value = spectrum_value
        
        return spectrum_value
    
    def get_results(self):
        """
        获取计算结果
        
        返回:
        tuple: (omega, spectrum_value) - 频率值和对应的谱密度值
        
        异常:
        ValueError: 当尚未进行计算时抛出
        """
        if self.spectrum_value is None:
            raise ValueError("尚未进行谱计算，请先调用calculate()方法")
        if self.omega is None:
            raise ValueError("频率值未设置，请先调用calculate()方法")
            
        return (self.omega, self.spectrum_value)


if __name__ == "__main__":
    # 示例1: 使用初始化参数计算特定频率点的谱密度
    pm = PiersonMoskowitzSpectrum(Hs=3.0, Tp=8.0)
    spectrum1 = pm.calculate(omega=0.5)  # 计算频率为0.5 rad/s处的谱密度S(ω)
    print(f"示例1 - 频率0.5 rad/s处的谱密度: {spectrum1}")
    
    # 示例2: 使用set_parameters方法设置参数后计算
    pm2 = PiersonMoskowitzSpectrum()
    pm2.set_parameters(Hs=4.0, Tp=10.0)
    spectrum2 = pm2.calculate(omega=0.6)  # 计算频率为0.6 rad/s处的谱密度S(ω)
    print(f"示例2 - 频率0.6 rad/s处的谱密度: {spectrum2}")
    
    # 示例3: 使用不同参数计算
    pm3 = PiersonMoskowitzSpectrum(Hs=2.5, Tp=7.0)
    spectrum3 = pm3.calculate(omega=0.7)  # 计算频率为0.7 rad/s处的谱密度S(ω)
    print(f"示例3 - 频率0.7 rad/s处的谱密度: {spectrum3}")
    
    # 示例4: 计算峰值频率处的谱密度
    pm4 = PiersonMoskowitzSpectrum(Hs=3.5, Tp=9.0)
    omega_p = 2 * math.pi / pm4.Tp  # 计算峰值频率
    spectrum4 = pm4.calculate(omega=omega_p)  # 计算峰值频率处的谱密度S(ωp)
    print(f"示例4 - 峰值频率{omega_p:.4f} rad/s处的谱密度: {spectrum4}")
    
    # 示例5: 获取计算结果
    pm5 = PiersonMoskowitzSpectrum(Hs=3.0, Tp=8.0)
    single_point_spectrum = pm5.calculate(omega=0.8)  # 计算频率为0.8 rad/s处的谱密度S(ω)
    omega, spectrum_value = pm5.get_results()  # 获取计算结果
    print(f"示例5 - 计算结果: 频率={omega}, 谱密度={spectrum_value}")