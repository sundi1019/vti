import numpy as np
import math
from enum import Enum


class AftBodyShape(Enum):
    """
    船舶尾部形状枚举类
    定义不同的尾部形状及其对应的C_stern系数
    """
    PRAM_WITH_GONDOLA = ("pram with gondola", -25)
    V_SHAPED_SECTIONS = ("V-shaped sections", -10)
    NORMAL_SECTIONS = ("normal sections", 0)
    U_SHAPED_SECTIONS = ("U-shaped sections", 10)

    def __init__(self, description, c_stern_value):
        self.description = description
        self.c_stern_value = c_stern_value
    
    @staticmethod
    def get_shape_by_description(description):
        """
        根据描述字符串查找对应的尾部形状枚举值
        
        Args:
            description (str): 尾部形状的描述字符串
            
        Returns:
            AftBodyShape: 对应的尾部形状枚举值
            
        Raises:
            ValueError: 当找不到匹配的尾部形状时抛出
        """
        for shape in AftBodyShape:
            if shape.description.lower() == description.lower():
                return shape
        raise ValueError(f"找不到描述为 '{description}' 的尾部形状")


class K_FactorCalculator:
    """
    形状因子k计算类
    
    基于Holtrop和Mennen方法计算船舶形状因子k
    该因子用于修正平板摩擦阻力R_F，考虑船舶几何形状的影响
    """

    def __init__(self, ship_params=None):
        """
        初始化船舶参数

        Args:
            ship_params (dict, optional): 船舶参数字典，包含：
                - L_WL: 水线长 (m)
                - B: 型宽 (m)
                - T: 型深 (m), 可选，如果未提供则需要同时提供T_a和T_f
                - T_a: 尾垂线吃水 (m), 如果T未提供则必须提供
                - T_f: 前垂线吃水 (m), 如果T未提供则必须提供
                - Delta: 型排水量 (m³) - 设计吃水下的体积排水量
                - A_m: 中剖面面积 (m²) - 设计吃水下的中剖面积
                - aft_body_shape: 尾部形状，可以是AftBodyShape枚举或字符串描述
                - V_W: 船舶通过水的速度 (m/s) - 可选，也可在计算时传入
        """
        # 默认值设置
        self.L_WL = None
        self.B = None
        self.T = None
        self.T_a = None
        self.T_f = None
        self.Delta = None
        self.A_m = None
        self.aft_body_shape = None
        self.V_W = None  # 船舶速度
        
        # 派生参数
        self.C_P = None  # 棱形系数
        
        # 结果存储
        self.results = {}
        
        # 如果提供了参数，则设置参数
        if ship_params is not None:
            self.set_parameters(ship_params)

    def set_parameters(self, ship_params):
        """
        设置船舶参数

        Args:
            ship_params (dict): 船舶参数字典
        """
        # 设置基本参数
        self.L_WL = ship_params['L_WL']
        self.B = ship_params['B']
        
        # 如果T未传递但T_a和T_f都存在，则自动计算T
        if 'T' not in ship_params and 'T_a' in ship_params and 'T_f' in ship_params:
            self.T_a = ship_params['T_a']
            self.T_f = ship_params['T_f']
            self.T = (self.T_a + self.T_f) / 2
        else:
            # 使用默认方式设置T
            self.T = ship_params.get('T', None)
            self.T_a = ship_params.get('T_a', self.T)
            self.T_f = ship_params.get('T_f', self.T)
        
        self.Delta = ship_params['Delta']
        self.A_m = ship_params['A_m']
        
        # 处理尾部形状参数，支持字符串转换为枚举
        aft_body_shape_param = ship_params['aft_body_shape']
        if isinstance(aft_body_shape_param, str):
            # 如果是字符串，尝试转换为枚举
            try:
                self.aft_body_shape = AftBodyShape.get_shape_by_description(aft_body_shape_param)
            except ValueError as e:
                raise ValueError(f"无效的尾部形状字符串: {e}")
        else:
            # 如果不是字符串，直接使用
            self.aft_body_shape = aft_body_shape_param
        
        self.V_W = ship_params.get('V_W')
        
        # 验证尾部形状参数
        if not isinstance(self.aft_body_shape, AftBodyShape):
            raise ValueError("aft_body_shape必须是AftBodyShape枚举类型或有效的字符串描述")
        
        # 计算派生参数
        self._calculate_derived_parameters()

        # 清空之前的计算结果
        self.results = {}

    def _calculate_derived_parameters(self):
        """
        计算派生参数
        包括棱形系数C_P等基础几何参数
        """
        # 计算棱形系数 C_P
        # C_P = 排水量 / (中剖面面积 × 水线长)
        self.C_P = self.Delta / (self.A_m * self.L_WL)

    def calculate(self):
        """
        计算形状因子k
        注意：V_W参数必须在初始化或通过set_parameters方法设置

        Returns:
            float: 形状因子k
        """
        # 验证参数是否已设置
        if self.L_WL is None or self.B is None:
            raise ValueError("船舶参数未设置，请先调用set_parameters方法")
        
        if self.V_W is None:
            raise ValueError("船舶速度V_W未设置，请在初始化时提供或通过set_parameters方法设置")
        
        # 计算Froude数
        Fr = self._calculate_froude_number()

        # 计算系数c14
        c14 = self._calculate_c14()

        # 计算浮心纵向位置
        l_CB = self._calculate_longitudinal_center_of_buoyancy(Fr)

        # 计算尾流长度L_R
        L_R = self._calculate_run_length(l_CB)

        # 计算形状因子k
        k = self._calculate_k_factor(c14, L_R)
        
        # 存储计算结果
        self.results = {
            'k': k,
            'V_W': self.V_W,
            'Fr': Fr,
            'c14': c14,
            'l_CB': l_CB,
            'L_R': L_R,
            'C_P': self.C_P
        }
        
        return k

    def _calculate_froude_number(self):
        """
        计算基于水线长的Froude数

        Returns:
            float: Froude数
        """
        g = 9.81  # 重力加速度 (m/s²)
        Fr = self.V_W / math.sqrt(g * self.L_WL)
        return Fr

    def _calculate_c14(self):
        """
        根据尾部形状计算系数c14

        Returns:
            float: 系数c14
        """
        C_stern = self.aft_body_shape.c_stern_value
        c14 = 1.0 + 0.011 * C_stern
        return c14

    def _calculate_longitudinal_center_of_buoyancy(self, Fr):
        """
        计算浮心纵向位置百分比

        Args:
            Fr (float): Froude数

        Returns:
            float: 浮心纵向位置 l_CB (%)
        """
        l_CB = -(0.44 * Fr - 0.094)
        return l_CB

    def _calculate_run_length(self, l_CB):
        """
        计算尾流长度L_R

        Args:
            l_CB (float): 浮心纵向位置 (%)

        Returns:
            float: 尾流长度L_R (m)
        """
        # 检查分母是否为零
        denominator = 4 * self.C_P - 1
        if abs(denominator) < 1e-10:
            # 使用简化公式
            L_R = self.L_WL * (1 - self.C_P)
        else:
            L_R = self.L_WL * (1 - self.C_P + (0.06 * self.C_P * l_CB) / denominator)

        return L_R

    def _calculate_k_factor(self, c14, L_R):
        """
        计算形状因子k

        Args:
            c14 (float): 系数c14
            L_R (float): 尾流长度 (m)

        Returns:
            float: 形状因子k
        """
        # 计算各个比值项
        B_LWL_ratio = self.B / self.L_WL
        T_LWL_ratio = self.T / self.L_WL
        LWL_LR_ratio = self.L_WL / L_R
        LWL3_Delta_ratio = (self.L_WL ** 3) / self.Delta

        # 检查(1 - C_P)是否为零或负数
        one_minus_Cp = 1 - self.C_P
        if one_minus_Cp <= 0:
            # 使用小的正值避免计算错误
            one_minus_Cp = max(one_minus_Cp, 1e-6)

        # 计算各项的幂次
        term1 = B_LWL_ratio ** 1.06806
        term2 = T_LWL_ratio ** 0.46106
        term3 = LWL_LR_ratio ** 0.121563
        term4 = LWL3_Delta_ratio ** 0.36486
        term5 = one_minus_Cp ** (-0.604247)

        # 计算最终的k值
        k = -0.07 + 0.487118 * c14 * term1 * term2 * term3 * term4 * term5

        return k

    def get_calculation_details(self):
        """
        获取计算过程的详细信息
        用于分析和验证计算过程
        注意：V_W参数必须在初始化或通过set_parameters方法设置

        Returns:
            dict: 包含所有中间计算结果的字典
        """
        # 先执行计算以确保所有结果都已生成
        self.calculate()
        
        # 获取所有中间结果
        Fr = self.results['Fr']
        c14 = self.results['c14']
        l_CB = self.results['l_CB']
        L_R = self.results['L_R']
        k = self.results['k']

        # 计算各个比值
        B_LWL_ratio = self.B / self.L_WL
        T_LWL_ratio = self.T / self.L_WL
        LWL_LR_ratio = self.L_WL / L_R
        LWL3_Delta_ratio = (self.L_WL ** 3) / self.Delta
        one_minus_Cp = 1 - self.C_P

        return {
            'input_parameters': {
                'L_WL': self.L_WL,
                'B': self.B,
                'T': self.T,
                'Delta': self.Delta,
                'A_m': self.A_m,
                'aft_body_shape': self.aft_body_shape.description,
                'V_W': self.V_W
            },
            'derived_parameters': {
                'C_P': self.C_P,
                'Fr': Fr,
                'C_stern': self.aft_body_shape.c_stern_value,
                'c14': c14,
                'l_CB_percent': l_CB,
                'L_R': L_R
            },
            'ratio_terms': {
                'B/L_WL': B_LWL_ratio,
                'T/L_WL': T_LWL_ratio,
                'L_WL/L_R': LWL_LR_ratio,
                'L_WL³/Δ': LWL3_Delta_ratio,
                '(1-C_P)': one_minus_Cp
            },
            'power_terms': {
                '(B/L_WL)^1.06806': B_LWL_ratio ** 1.06806,
                '(T/L_WL)^0.46106': T_LWL_ratio ** 0.46106,
                '(L_WL/L_R)^0.121563': LWL_LR_ratio ** 0.121563,
                '(L_WL³/Δ)^0.36486': LWL3_Delta_ratio ** 0.36486,
                '(1-C_P)^-0.604247': one_minus_Cp ** (-0.604247)
            },
            'final_result': {
                'form_factor_k': k
            }
        }

    def validate_parameters(self):
        """
        验证船舶参数的合理性

        Returns:
            tuple: (是否有效, 验证信息)
        """
        errors = []
        warnings = []

        # 检查基本几何参数
        if self.L_WL is None:
            errors.append("水线长 L_WL 未设置")
        elif self.L_WL <= 0:
            errors.append(f"水线长 L_WL = {self.L_WL} 必须为正数")

        if self.B is None:
            errors.append("船宽 B 未设置")
        elif self.B <= 0:
            errors.append(f"船宽 B = {self.B} 必须为正数")

        # 检查T或T_a/T_f参数
        if self.T is None and (self.T_a is None or self.T_f is None):
            errors.append("必须设置T或同时设置T_a和T_f")
        elif self.T is not None and self.T <= 0:
            errors.append(f"吃水 T = {self.T} 必须为正数")
        elif self.T is None and (self.T_a <= 0 or self.T_f <= 0):
            errors.append(f"吃水 T_a = {self.T_a} 和 T_f = {self.T_f} 必须为正数")
        
        # 检查船舶速度V_W
        if self.V_W is None:
            errors.append("船舶速度 V_W 未设置")
        elif self.V_W <= 0:
            errors.append(f"船舶速度 V_W = {self.V_W} 必须为正数")

        if self.Delta is None:
            errors.append("排水量 Delta 未设置")
        elif self.Delta <= 0:
            errors.append(f"排水量 Delta = {self.Delta} 必须为正数")

        if self.A_m is None:
            errors.append("中剖面面积 A_m 未设置")
        elif self.A_m <= 0:
            errors.append(f"中剖面面积 A_m = {self.A_m} 必须为正数")

        if self.aft_body_shape is None:
            errors.append("尾部形状 aft_body_shape 未设置")
        elif not isinstance(self.aft_body_shape, AftBodyShape):
            errors.append("尾部形状必须是AftBodyShape枚举类型")

        # 如果已设置参数，检查几何比例
        if not errors:
            L_B_ratio = self.L_WL / self.B
            if L_B_ratio < 3 or L_B_ratio > 15:
                warnings.append(f"长宽比 L_WL/B = {L_B_ratio:.2f} 超出常见范围 [3, 15]")

            B_T_ratio = self.B / self.T
            if B_T_ratio < 2 or B_T_ratio > 5:
                warnings.append(f"宽深比 B/T = {B_T_ratio:.2f} 超出常见范围 [2, 5]")

            # 检查棱形系数
            if not (0.5 <= self.C_P <= 0.9):
                warnings.append(f"棱形系数 C_P = {self.C_P:.3f} 超出常见范围 [0.5, 0.9]")

        if errors:
            return False, f"错误: {'; '.join(errors)}"
        elif warnings:
            return True, f"警告: {'; '.join(warnings)}"
        else:
            return True, "所有参数验证通过"

    @staticmethod
    def get_aft_body_shape_options():
        """
        获取所有可用的尾部形状选项

        Returns:
            dict: 尾部形状描述与对应的枚举值的映射
        """
        return {shape.description: shape for shape in AftBodyShape}


# 使用示例
if __name__ == "__main__":
    # 定义船舶参数示例（包含T参数和V_W参数，使用字符串形式的尾部形状）
    ship_params = {
        'L_WL': 180.0,  # 水线长 (m)
        'B': 28.0,  # 船宽 (m)
        'T': 8.5,  # 平均吃水 (m)
        'T_a': 9.0,  # 尾部吃水 (m)
        'T_f': 8.0,  # 前部吃水 (m)
        'Delta': 25000.0,  # 排水量 (m³)
        'A_m': 220.0,  # 中剖面面积 (m²)
        'aft_body_shape': "normal sections",  # 尾部形状（字符串形式）
        'V_W': 10.0  # 船舶速度 (m/s)
    }
    
    # 定义船舶参数示例（不包含T参数，使用T_a和T_f，但包含V_W参数）
    ship_params_without_T = {
        'L_WL': 180.0,  # 水线长 (m)
        'B': 28.0,  # 船宽 (m)
        'T_a': 9.0,  # 尾部吃水 (m)
        'T_f': 8.0,  # 前部吃水 (m)
        'Delta': 25000.0,  # 排水量 (m³)
        'A_m': 220.0,  # 中剖面面积 (m²)
        'aft_body_shape': "normal sections",  # 尾部形状（字符串形式）
        'V_W': 10.0  # 船舶速度 (m/s)
    }
    
    # 定义船舶参数示例（使用不同大小写形式的尾部形状字符串）
    ship_params_with_uppercase_shape = {
        'L_WL': 180.0,  # 水线长 (m)
        'B': 28.0,  # 船宽 (m)
        'T': 8.5,  # 平均吃水 (m)
        'Delta': 25000.0,  # 排水量 (m³)
        'A_m': 220.0,  # 中剖面面积 (m²)
        'aft_body_shape': "U-SHAPED SECTIONS",  # 大写形式的尾部形状字符串
        'V_W': 10.0  # 船舶速度 (m/s)
    }

    print("示例1: 初始化时设置船舶参数，计算特定速度下的k值")
    # 创建计算对象
    k_calculator = K_FactorCalculator(ship_params)

    # 验证参数
    is_valid, message = k_calculator.validate_parameters()
    print(f"参数验证: {message}")

    if is_valid:
        # 计算k值（已在初始化参数中提供V_W）
        k = k_calculator.calculate()
        print(f"速度{ship_params['V_W']}m/s时的形状因子k: {k:.6f}")
    
    print("\n示例2: 计算多组速度下的形状因子")
    print("\n形状因子计算结果:")
    print("速度(m/s) | Froude数 | 形状因子k")
    print("-" * 35)
    
    # 计算不同速度下的k值
    test_speeds = [5, 10, 15, 20]  # m/s

    for speed in test_speeds:
        try:
            # 通过set_parameters更新V_W参数
            temp_params = ship_params.copy()
            temp_params['V_W'] = speed
            k_calculator.set_parameters(temp_params)
            
            # 计算k值
            k = k_calculator.calculate()
            Fr = k_calculator.results['Fr']
            print(f"{speed:8.1f} | {Fr:8.4f} | {k:10.6f}")
        except Exception as e:
            print(f"{speed:8.1f} | 计算错误: {e}")
    
    print("-" * 45)