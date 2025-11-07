import numpy as np
import math
from Cx import WindResistanceCoefficientCalculator


class WindResistanceCalculator:
    """
    风阻力计算类
    
    基于DNV-RP-0675标准计算船舶风阻力
    使用公式: R_wind = 0.5*Cx(ψ_rel)*ρ_air*V_rel²*A_VS - 0.5*Cx(0)*ρ_air*V_G²*A_VS
    """

    def __init__(self, ship_params=None):
        """
        初始化船舶参数和风况参数

        Args:
            ship_params (dict, optional): 船舶参数字典，包含：
                - L_OA: 船舶总长 (m)
                - B: 船宽 (m)
                - A_OD: 上甲板以上上层建筑侧向投影面积 (m²)
                - A_XV: 水线以上包括上层建筑的横向投影面积 (m²)
                - A_LV: 水线以上包括上层建筑的侧向投影面积 (m²)
                - C_MC: 船中到侧向投影面积A_LV重心的水平距离 (m，向前为正)
                - H_BR: 上层建筑(驾驶台等)顶部高度 (m)
                - H_C: 从水线到侧向投影面积A_LV重心的高度 (m)
                - A_VS: 水线以上包括上层建筑在内的横向投影面积 (m²)
                - mu: 平滑范围角度 (度，通常为10度)
                - psi_rel: 相对风向角度 (度)，0度表示顶风
                - V_rel: 相对风速 (m/s)
                - V_G: 船舶对地速度 (m/s)
        """
        # 默认值设置
        self.L_OA = None
        self.B = None
        self.A_OD = None
        self.A_XV = None
        self.A_LV = None
        self.A_VS = None  # 横向投影面积，用于风阻力计算
        self.C_MC = None
        self.H_BR = None
        self.H_C = None
        self.mu = None
        
        # 风况参数
        self.psi_rel = None
        self.V_rel = None
        self.V_G = None
        
        # Cx计算器实例
        self.cx_calculator = None
        
        # 物理常数
        self.rho_air = 1.225  # 空气密度 (kg/m³)，在15°C时
        
        # 结果存储
        self.results = {}
        
        # 初始化Cx计算器
        self.cx_calculator = None
        
        # 如果提供了参数，则设置参数
        if ship_params is not None:
            self.set_parameters(ship_params)

    def set_parameters(self, ship_params):
        """
        设置船舶参数和风况参数

        Args:
            ship_params (dict): 参数字典，包含船舶参数和风况参数
        """
        # 设置船舶参数
        self.L_OA = ship_params['L_OA']
        self.B = ship_params['B']
        self.A_OD = ship_params['A_OD']
        self.A_XV = ship_params['A_XV']
        self.A_LV = ship_params['A_LV']
        self.A_VS = ship_params.get('A_VS', ship_params['A_XV'])  # 如果没有提供A_VS，使用A_XV作为默认值
        self.C_MC = ship_params['C_MC']
        self.H_BR = ship_params['H_BR']
        self.H_C = ship_params['H_C']
        self.mu = ship_params.get('mu', 10.0)  # 默认平滑范围10度
        
        # 设置风况参数
        self.psi_rel = ship_params.get('psi_rel')
        self.V_rel = ship_params.get('V_rel')
        self.V_G = ship_params.get('V_G')

        # 设置Cx计算器参数（仅包含船舶参数）
        cx_params = {
            'L_OA': self.L_OA,
            'B': self.B,
            'A_OD': self.A_OD,
            'A_XV': self.A_XV,
            'A_LV': self.A_LV,
            'C_MC': self.C_MC,
            'H_BR': self.H_BR,
            'H_C': self.H_C,
            'mu': self.mu
        }
        
        self.cx_calculator = WindResistanceCoefficientCalculator()
        self.cx_calculator.set_parameters(cx_params)
        
        # 清空之前的计算结果
        self.results = {}
        
        # 统一校验所有输入参数
        self._validate_parameters()

    def calculate(self):
        """
        计算风阻力

        Returns:
            float: 风阻力 R_wind (N)
        """
        print("\n=== 开始计算风阻力 R_wind ===")
        
        # 验证参数是否已设置
        if self.L_OA is None or self.B is None:
            raise ValueError("船舶参数未设置，请先调用set_parameters方法")
        
        if self.cx_calculator is None:
            raise ValueError("Cx计算器未初始化，请先调用set_parameters方法")
        
        if self.psi_rel is None or self.V_rel is None or self.V_G is None:
            raise ValueError("风况参数未设置，请在初始化时提供或在外部设置")
        
        # 验证输入参数
        if not isinstance(self.psi_rel, (int, float)):
            raise TypeError("psi_rel必须是单个数值")
        if not isinstance(self.V_rel, (int, float)):
            raise TypeError("V_rel必须是单个数值")
        if not isinstance(self.V_G, (int, float)):
            raise TypeError("V_G必须是单个数值")
        
        # 打印输入参数
        print(f"\n1. 输入参数:")
        print(f"   - 船舶总长(L_OA): {self.L_OA:.2f} m")
        print(f"   - 船宽(B): {self.B:.2f} m")
        print(f"   - 横向投影面积(A_VS): {self.A_VS:.2f} m²")
        print(f"   - 空气密度(ρ_air): {self.rho_air:.3f} kg/m³")
        print(f"   - 相对风向角度(ψ_rel): {self.psi_rel:.1f}°")
        print(f"   - 相对风速(V_rel): {self.V_rel:.2f} m/s")
        print(f"   - 对地速度(V_G): {self.V_G:.2f} m/s")
        
        # 计算Cx(ψ_rel)
        print(f"\n2. 计算风阻力系数:")
        Cx_psi_rel = self.cx_calculator.calculate(self.psi_rel)
        print(f"   - Cx(ψ_rel={self.psi_rel:.1f}°) = {Cx_psi_rel:.6f}")
        
        # 计算Cx(0) - 顶风阻力系数
        Cx_0 = self.cx_calculator.calculate(0)
        print(f"   - Cx(0°) [顶风阻力系数] = {Cx_0:.6f}")
        
        # 计算风阻力
        print(f"\n3. 计算风阻力公式: R_wind = 0.5*Cx(ψ_rel)*ρ_air*V_rel²*A_VS - 0.5*Cx(0)*ρ_air*V_G²*A_VS")
        
        # 计算第一项
        term1 = 0.5 * Cx_psi_rel * self.rho_air * (self.V_rel ** 2) * self.A_VS
        print(f"   - 第一项计算: 0.5 * {Cx_psi_rel:.6f} * {self.rho_air:.3f} * ({self.V_rel:.2f})² * {self.A_VS:.2f} = {term1:.2f} N")
        
        # 计算第二项
        term2 = 0.5 * Cx_0 * self.rho_air * (self.V_G ** 2) * self.A_VS
        print(f"   - 第二项计算: 0.5 * {Cx_0:.6f} * {self.rho_air:.3f} * ({self.V_G:.2f})² * {self.A_VS:.2f} = {term2:.2f} N")
        
        # 计算最终风阻力
        R_wind = term1 - term2
        print(f"\n4. 最终风阻力计算:")
        print(f"   - R_wind = 第一项 - 第二项 = {term1:.2f} - {term2:.2f} = {R_wind:.2f} N")
        
        # 存储计算结果
        self.results = {
            'R_wind': R_wind,
            'psi_rel': self.psi_rel,
            'V_rel': self.V_rel,
            'V_G': self.V_G,
            'Cx_psi_rel': Cx_psi_rel,
            'Cx_0': Cx_0,
            'rho_air': self.rho_air,
            'A_VS': self.A_VS,
            'term1': term1,
            'term2': term2
        }
        
        print(f"\n=== 风阻力计算完成 ===")
        
        return R_wind





    def _validate_parameters(self):
        """
        统一校验所有输入参数
        """
        # 检查必要参数是否为正数
        required_positive = ['L_OA', 'B', 'A_LV', 'H_BR', 'A_VS']
        for param in required_positive:
            value = getattr(self, param)
            if value is None:
                raise ValueError(f"{param} 未设置")
            elif value <= 0:
                raise ValueError(f"{param} 必须为正数，当前值: {value}")

        # 检查速度参数的合理性
        if self.V_rel is not None and self.V_rel < 0:
            raise ValueError(f"相对风速(V_rel)不能为负数，当前值: {self.V_rel}")
        
        if self.V_G is not None and self.V_G < 0:
            raise ValueError(f"对地速度(V_G)不能为负数，当前值: {self.V_G}")
        
        # 检查角度参数的合理性
        if self.psi_rel is not None and (self.psi_rel < 0 or self.psi_rel > 360):
            raise ValueError(f"相对风向角度(psi_rel)应在0-360度之间，当前值: {self.psi_rel}")
        
        if self.mu is not None and (self.mu < 0 or self.mu > 90):
            raise ValueError(f"平滑范围角度(mu)应在0-90度之间，当前值: {self.mu}")

        # 检查Cx计算器参数
        if self.cx_calculator is not None:
            # 注意：需要确认Cx计算器的验证方法是否也已修改为抛出异常的方式
            # 如果未修改，则这里可能需要适配原有的返回值形式
            try:
                # 尝试调用Cx计算器的参数验证方法
                if hasattr(self.cx_calculator, '_validate_parameters'):
                    self.cx_calculator._validate_parameters()  # 调用带下划线的验证方法
                elif hasattr(self.cx_calculator, 'validate_parameters'):
                    # 兼容原有接口，但将返回值转换为异常
                    valid, message = self.cx_calculator.validate_parameters()
                    if not valid:
                        raise ValueError(f"Cx计算器参数验证失败: {message}")
            except Exception as e:
                raise ValueError(f"Cx计算器参数验证错误: {str(e)}")
    
    def validate_parameters(self):
        """
        兼容性方法，调用新的_validate_parameters方法
        
        Returns:
            tuple: (是否有效, 验证信息)
        """
        try:
            self._validate_parameters()
            return True, "所有参数验证通过"
        except ValueError as e:
            return False, str(e)


# 使用示例
if __name__ == "__main__":
    # 定义船舶参数示例
    ship_params = {
        'L_OA': 180.0,  # 船舶总长 (m)
        'B': 28.0,  # 船宽 (m)
        'A_OD': 850.0,  # 上甲板以上侧向投影面积 (m²)
        'A_XV': 1200.0,  # 横向投影面积 (m²)
        'A_LV': 2800.0,  # 侧向投影面积 (m²)
        'A_VS': 1200.0,  # 横向投影面积，用于风阻力计算 (m²)
        'C_MC': 5.0,  # 到重心的距离 (m)
        'H_BR': 45.0,  # 上层建筑高度 (m)
        'H_C': 15.0,  # 重心高度 (m)
        'mu': 10.0  # 平滑角度 (度)
    }

    print("示例1: 初始化时设置船舶参数和风况参数")
    # 直接设置完整参数
    ship_params_90 = {
        'L_OA': 180.0,  # 船舶总长 (m)
        'B': 28.0,  # 船宽 (m)
        'A_OD': 850.0,  # 上甲板以上侧向投影面积 (m²)
        'A_XV': 1200.0,  # 横向投影面积 (m²)
        'A_LV': 2800.0,  # 侧向投影面积 (m²)
        'A_VS': 1200.0,  # 横向投影面积，用于风阻力计算 (m²)
        'C_MC': 5.0,  # 到重心的距离 (m)
        'H_BR': 45.0,  # 上层建筑高度 (m)
        'H_C': 15.0,  # 重心高度 (m)
        'mu': 10.0,  # 平滑角度 (度)
        'psi_rel': 90,  # 横风
        'V_rel': 10,    # 相对风速10m/s
        'V_G': 5        # 船速5m/s
    }
    
    # 创建计算对象并设置所有参数
    wind_res_calc = WindResistanceCalculator(ship_params=ship_params_90)

    R_wind = wind_res_calc.calculate()
    
    # 打印计算结果
    print(f"\n示例1计算结果:")
    print(f"风阻力R_wind = {R_wind:.2f} N")