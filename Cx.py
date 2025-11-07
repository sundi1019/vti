import math


class WindResistanceCoefficientCalculator:
    """
    风阻系数计算类 (按照RAW格式重写)
    
    基于ISO 15016标准计算不同相对风向下的风阻系数C_AA
    适用于船舶在风载荷作用下的阻力计算
    """

    def __init__(self, ship_params=None):
        """
        初始化船舶参数

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
                - mu: 平滑范围角度 (度，通常为10度)
        """
        # 默认值设置
        self.L_OA = None
        self.B = None
        self.A_OD = None
        self.A_XV = None
        self.A_LV = None
        self.C_MC = None
        self.H_BR = None
        self.H_C = None
        self.mu = None
        
        # 结果存储
        self.results = {}
        
        # 如果提供了参数，则设置参数
        if ship_params is not None:
            self.set_parameters(ship_params)

    def set_parameters(self, ship_params):
        """
        设置船舶参数

        Args:
            ship_params (dict): 船舶参数字典，包含必要的船舶参数
        """
        self.L_OA = ship_params['L_OA']
        self.B = ship_params['B']
        self.A_OD = ship_params['A_OD']
        self.A_XV = ship_params['A_XV']
        self.A_LV = ship_params['A_LV']
        self.C_MC = ship_params['C_MC']
        self.H_BR = ship_params['H_BR']
        self.H_C = ship_params['H_C']
        self.mu = ship_params.get('mu', 10.0)  # 默认平滑范围10度

        # 初始化回归系数表
        self._initialize_coefficients()
        
        # 清空之前的计算结果
        self.results = {}

    def _initialize_coefficients(self):
        """
        初始化ISO 15016标准中的无量纲回归参数
        根据Table A-1设置β_ij, δ_ij, ε_ij系数
        """
        # β系数 (beta coefficients)
        self.beta = {
            # i=1的系数 (0° ≤ ψ_WR < 90°)
            (1, 0): 0.922,
            (1, 1): -0.507,
            (1, 2): -1.162,
            # i=2的系数 (90° < ψ_WR ≤ 180°)
            (2, 0): -0.018,
            (2, 1): 5.091,
            (2, 2): -10.367,
            (2, 3): 3.011,
            (2, 4): 0.341
        }

        # δ系数 (delta coefficients)
        self.delta = {
            # i=1的系数 (0° ≤ ψ_WR < 90°)
            (1, 0): -0.458,
            (1, 1): -3.245,
            (1, 2): 2.313,
            # i=2的系数 (90° < ψ_WR ≤ 180°)
            (2, 0): 1.901,
            (2, 1): -12.727,
            (2, 2): -24.407,
            (2, 3): 40.310,
            (2, 4): 5.481
        }

        # ε系数 (epsilon coefficients)
        self.epsilon = {
            # i=1的系数 (0° ≤ ψ_WR < 90°)
            (1, 0): 0.585,
            (1, 1): 0.906,
            (1, 2): -3.239,
            # i=2的系数 (90° < ψ_WR ≤ 180°)
            (2, 0): 0.314,
            (2, 1): 1.117
        }

    def calculate(self, psi_WR):
        """
        计算给定相对风向下的风阻系数C_AA

        Args:
            psi_WR (float): 相对风向角度 (度)，0度表示顶风

        Returns:
            float: 风阻系数C_AA
        """
        print("\n=== 开始计算风阻系数C_AA ===")
        
        # 打印输入参数
        print(f"\n1. 输入参数:")
        print(f"   - 输入相对风向角度(原始值): {psi_WR:.1f}°")
        print(f"   - 船舶总长(L_OA): {self.L_OA:.2f} m")
        print(f"   - 船宽(B): {self.B:.2f} m")
        if self.A_OD is not None:
            print(f"   - 上甲板以上上层建筑侧向投影面积(A_OD): {self.A_OD:.2f} m²")
        if self.A_XV is not None:
            print(f"   - 水线以上横向投影面积(A_XV): {self.A_XV:.2f} m²")
        if self.A_LV is not None:
            print(f"   - 水线以上侧向投影面积(A_LV): {self.A_LV:.2f} m²")
        if self.mu is not None:
            print(f"   - 平滑范围角度(mu): {self.mu:.1f}°")
        
        # 验证参数是否已设置
        if self.L_OA is None or self.B is None:
            raise ValueError("船舶参数未设置，请先调用set_parameters方法")
        
        # 验证输入参数
        if not isinstance(psi_WR, (int, float)):
            raise TypeError("psi_WR必须是单个数值")
        
        # 确保角度在0-180度范围内
        normalized_angle = self._normalize_angle(psi_WR)
        print(f"\n2. 角度标准化处理:")
        print(f"   - 标准化后的相对风向角度: {normalized_angle:.1f}°")
        
        # 计算风阻系数
        print(f"\n3. 计算风阻系数:")
        if 0 <= normalized_angle < 90:
            print(f"   - 使用0°到90°范围的计算公式")
            C_AA = self._calculate_coefficient_0_to_90(normalized_angle)
        elif normalized_angle == 90:
            print(f"   - 使用90°点的特殊计算公式（平滑插值）")
            C_AA = self._calculate_coefficient_at_90()
        elif 90 < normalized_angle <= 180:
            print(f"   - 使用90°到180°范围的计算公式")
            C_AA = self._calculate_coefficient_90_to_180(normalized_angle)
        else:
            raise ValueError(f"相对风向角 {normalized_angle} 超出有效范围 [0, 180]度")
        
        print(f"\n4. 计算结果:")
        print(f"   - 风阻系数C_AA: {C_AA:.6f}")
        
        # 存储计算结果
        self.results = {
            'C_AA': C_AA,
            'psi_WR': normalized_angle,
            'L_OA': self.L_OA,
            'B': self.B,
            'A_OD': self.A_OD,
            'A_XV': self.A_XV,
            'A_LV': self.A_LV
        }
        
        print(f"\n=== 风阻系数计算完成 ===")
        
        return C_AA

    def _normalize_angle(self, angle):
        """
        将角度标准化到0-180度范围

        Args:
            angle (float): 输入角度

        Returns:
            float: 标准化后的角度
        """
        # 将角度限制在0-360度范围
        angle = angle % 360

        # 如果角度大于180度，使用对称性转换
        if angle > 180:
            angle = 360 - angle

        return angle

    def _calculate_coefficient_0_to_90(self, psi_WR):
        """
        计算0° ≤ ψ_WR < 90°范围内的风阻系数

        Args:
            psi_WR (float): 相对风向角度 (度)

        Returns:
            float: 风阻系数C_AA
        """
        print(f"   \n   详细计算过程 (0°-90°范围):")
        
        # 转换为弧度
        psi_rad = math.radians(psi_WR)
        print(f"   - 相对风向角度(弧度): {psi_rad:.6f} rad")

        # 计算C_LF系数
        C_LF = self._calculate_C_LF_range_1()
        print(f"   - 计算C_LF系数: {C_LF:.6f}")

        # 计算C_XLI系数
        C_XLI = self._calculate_C_XLI_range_1()
        print(f"   - 计算C_XLI系数: {C_XLI:.6f}")

        # 计算C_ALF系数
        C_ALF = self._calculate_C_ALF_range_1()
        print(f"   - 计算C_ALF系数: {C_ALF:.6f}")

        # 计算主公式各项
        cos_psi = math.cos(psi_rad)
        sin_psi = math.sin(psi_rad)
        print(f"   - cos(ψ_WR): {cos_psi:.6f}, sin(ψ_WR): {sin_psi:.6f}")
        
        term1 = C_LF * cos_psi
        print(f"   - 项1: C_LF * cos(ψ_WR) = {C_LF:.6f} * {cos_psi:.6f} = {term1:.6f}")
        
        term2_part1 = sin_psi - 0.5 * sin_psi * (cos_psi ** 2)
        term2 = C_XLI * term2_part1 * sin_psi * cos_psi
        print(f"   - 项2: C_XLI * [sin(ψ_WR)-0.5*sin(ψ_WR)*cos²(ψ_WR)] * sin(ψ_WR) * cos(ψ_WR) = {term2:.6f}")
        
        term3 = C_ALF * sin_psi * (cos_psi ** 3)
        print(f"   - 项3: C_ALF * sin(ψ_WR) * cos³(ψ_WR) = {C_ALF:.6f} * {sin_psi:.6f} * {cos_psi**3:.6f} = {term3:.6f}")
        
        total = term1 + term2 + term3
        print(f"   - 总和: {term1:.6f} + {term2:.6f} + {term3:.6f} = {total:.6f}")

        return total

    def _calculate_coefficient_90_to_180(self, psi_WR):
        """
        计算90° < ψ_WR ≤ 180°范围内的风阻系数

        Args:
            psi_WR (float): 相对风向角度 (度)

        Returns:
            float: 风阻系数C_AA
        """
        print(f"   \n   详细计算过程 (90°-180°范围):")
        
        # 转换为弧度
        psi_rad = math.radians(psi_WR)
        print(f"   - 相对风向角度(弧度): {psi_rad:.6f} rad")

        # 计算C_LF系数
        C_LF = self._calculate_C_LF_range_2()
        print(f"   - 计算C_LF系数: {C_LF:.6f}")

        # 计算C_XLI系数
        C_XLI = self._calculate_C_XLI_range_2()
        print(f"   - 计算C_XLI系数: {C_XLI:.6f}")

        # 计算C_ALF系数
        C_ALF = self._calculate_C_ALF_range_2()
        print(f"   - 计算C_ALF系数: {C_ALF:.6f}")

        # 计算主公式各项
        cos_psi = math.cos(psi_rad)
        sin_psi = math.sin(psi_rad)
        print(f"   - cos(ψ_WR): {cos_psi:.6f}, sin(ψ_WR): {sin_psi:.6f}")
        
        term1 = C_LF * cos_psi
        print(f"   - 项1: C_LF * cos(ψ_WR) = {C_LF:.6f} * {cos_psi:.6f} = {term1:.6f}")
        
        term2_part1 = sin_psi - 0.5 * sin_psi * (cos_psi ** 2)
        term2 = C_XLI * term2_part1 * sin_psi * cos_psi
        print(f"   - 项2: C_XLI * [sin(ψ_WR)-0.5*sin(ψ_WR)*cos²(ψ_WR)] * sin(ψ_WR) * cos(ψ_WR) = {term2:.6f}")
        
        term3 = C_ALF * sin_psi * (cos_psi ** 3)
        print(f"   - 项3: C_ALF * sin(ψ_WR) * cos³(ψ_WR) = {C_ALF:.6f} * {sin_psi:.6f} * {cos_psi**3:.6f} = {term3:.6f}")
        
        total = term1 + term2 + term3
        print(f"   - 总和: {term1:.6f} + {term2:.6f} + {term3:.6f} = {total:.6f}")

        return total

    def _calculate_coefficient_at_90(self):
        """
        计算ψ_WR = 90°时的风阻系数
        使用平滑插值方法

        Returns:
            float: 90度时的风阻系数C_AA
        """
        print(f"   \n   详细计算过程 (90°平滑插值):")
        print(f"   - 平滑范围角度(mu): {self.mu:.1f}°")
        
        # 计算90-μ和90+μ处的系数
        angle_minus = 90 - self.mu
        angle_plus = 90 + self.mu
        print(f"   - 计算ψ_WR = {angle_minus:.1f}°和{angle_plus:.1f}°处的系数")
        C_AA_minus = self._calculate_coefficient_0_to_90(angle_minus)
        C_AA_plus = self._calculate_coefficient_90_to_180(angle_plus)

        # 使用平滑公式
        C_AA_90 = 0.5 * (C_AA_minus + C_AA_plus)
        print(f"   - 平滑插值: C_AA(90°) = 0.5 * ({C_AA_minus:.6f} + {C_AA_plus:.6f}) = {C_AA_90:.6f}")

        return C_AA_90

    def _calculate_C_LF_range_1(self):
        """
        计算0° ≤ ψ_WR < 90°范围的C_LF系数

        Returns:
            float: C_LF系数
        """
        term1 = self.beta[(1, 0)]
        term2 = self.beta[(1, 1)] * (self.A_LV / (self.L_OA * self.B))
        term3 = self.beta[(1, 2)] * (self.C_MC / self.L_OA)

        return term1 + term2 + term3

    def _calculate_C_LF_range_2(self):
        """
        计算90° < ψ_WR ≤ 180°范围的C_LF系数

        Returns:
            float: C_LF系数
        """
        term1 = self.beta[(2, 0)]
        term2 = self.beta[(2, 1)] * (self.B / self.L_OA)
        term3 = self.beta[(2, 2)] * (self.H_C / self.L_OA)
        term4 = self.beta[(2, 3)] * (self.A_OD / (self.L_OA ** 2))
        term5 = self.beta[(2, 4)] * (self.A_XV / (self.B ** 2))

        return term1 + term2 + term3 + term4 + term5

    def _calculate_C_XLI_range_1(self):
        """
        计算0° ≤ ψ_WR < 90°范围的C_XLI系数

        Returns:
            float: C_XLI系数
        """
        term1 = self.delta[(1, 0)]
        term2 = self.delta[(1, 1)] * (self.A_LV / (self.L_OA * self.H_BR))
        term3 = self.delta[(1, 2)] * (self.A_XV / (self.B * self.H_BR))

        return term1 + term2 + term3

    def _calculate_C_XLI_range_2(self):
        """
        计算90° < ψ_WR ≤ 180°范围的C_XLI系数

        Returns:
            float: C_XLI系数
        """
        term1 = self.delta[(2, 0)]
        term2 = self.delta[(2, 1)] * (self.A_LV / (self.L_OA * self.H_BR))
        term3 = self.delta[(2, 2)] * (self.A_XV / self.A_LV) if self.A_LV != 0 else 0
        term4 = self.delta[(2, 3)] * (self.B / self.L_OA)
        term5 = self.delta[(2, 4)] * (self.A_XV / (self.B * self.H_BR))

        return term1 + term2 + term3 + term4 + term5

    def _calculate_C_ALF_range_1(self):
        """
        计算0° ≤ ψ_WR < 90°范围的C_ALF系数

        Returns:
            float: C_ALF系数
        """
        term1 = self.epsilon[(1, 0)]
        term2 = self.epsilon[(1, 1)] * (self.A_OD / self.A_LV) if self.A_LV != 0 else 0
        term3 = self.epsilon[(1, 2)] * (self.B / self.L_OA)

        return term1 + term2 + term3

    def _calculate_C_ALF_range_2(self):
        """
        计算90° < ψ_WR ≤ 180°范围的C_ALF系数

        Returns:
            float: C_ALF系数
        """
        term1 = self.epsilon[(2, 0)]
        term2 = self.epsilon[(2, 1)] * (self.A_OD / self.A_LV) if self.A_LV != 0 else 0

        return term1 + term2

    def validate_parameters(self):
        """
        验证船舶参数的合理性

        Returns:
            tuple: (是否有效, 验证信息)
        """
        errors = []
        warnings = []

        # 检查必要参数是否为正数
        required_positive = ['L_OA', 'B', 'A_LV', 'H_BR']
        for param in required_positive:
            value = getattr(self, param)
            if value is None:
                errors.append(f"{param} 未设置")
            elif value <= 0:
                errors.append(f"{param} = {value} 必须为正数")

        # 检查面积参数的合理性
        if self.A_OD is not None and self.A_OD < 0:
            errors.append(f"A_OD = {self.A_OD} 不能为负数")

        if self.A_XV is not None and self.A_XV < 0:
            errors.append(f"A_XV = {self.A_XV} 不能为负数")

        # 检查高度参数
        if self.H_C is not None and self.H_C < 0:
            warnings.append(f"H_C = {self.H_C} 为负数，请确认测量基准")

        # 检查几何比例的合理性
        if self.L_OA is not None and self.B is not None:
            L_B_ratio = self.L_OA / self.B
            if L_B_ratio < 3 or L_B_ratio > 15:
                warnings.append(f"L_OA/B = {L_B_ratio:.2f} 超出常见船舶范围 [3, 15]")

        # 检查平滑角度
        if self.mu is not None and not (0 < self.mu < 45):
            warnings.append(f"平滑角度 μ = {self.mu}° 建议在0-45度范围内")

        if errors:
            return False, f"错误: {'; '.join(errors)}"
        elif warnings:
            return True, f"警告: {'; '.join(warnings)}"
        else:
            return True, "所有参数验证通过"


# 使用示例
if __name__ == "__main__":
    # 定义船舶参数示例
    ship_params = {
        'L_OA': 180.0,  # 船舶总长 (m)
        'B': 28.0,  # 船宽 (m)
        'A_OD': 850.0,  # 上甲板以上侧向投影面积 (m²)
        'A_XV': 1200.0,  # 横向投影面积 (m²)
        'A_LV': 2800.0,  # 侧向投影面积 (m²)
        'C_MC': 5.0,  # 到重心的距离 (m)
        'H_BR': 45.0,  # 上层建筑高度 (m)
        'H_C': 15.0,  # 重心高度 (m)
        'mu': 10.0  # 平滑角度 (度)
    }

    # 创建计算对象
    wind_calc = WindResistanceCoefficientCalculator()
    
    # 设置参数
    wind_calc.set_parameters(ship_params)

    # 验证参数
    is_valid, message = wind_calc.validate_parameters()
    print(f"参数验证: {message}")

    if is_valid:
        # 计算不同风向下的系数
        test_angles = [0, 30, 60, 90, 120, 150, 180]

        print("\n风阻系数计算结果:")
        print("相对风向(度) | 风阻系数C_AA")
        print("-" * 30)

        for angle in test_angles:
            coeff = wind_calc.calculate(angle)
            print(f"{angle:8.0f}     | {coeff:10.6f}")