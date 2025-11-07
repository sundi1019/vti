import numpy as np
import math


class WaveAddedResistanceCalculator:
    """
    波浪附加阻力计算类
    
    基于经验关系计算波浪附加阻力的RAO（Response Amplitude Operator）
    包括运动引起的附加阻力(R_AWM)和反射引起的附加阻力(R_AWR)
    """

    def __init__(self, ship_params=None):
        """
        初始化船舶参数

        Args:
            ship_params (dict, optional): 船舶参数字典，包含：
                - L_PP: 垂线间长 (m)
                - B: 船宽 (m)
                - C_B: 方形系数
                - T_f: 前垂线吃水 (m)
                - T_a: 后垂线吃水 (m)
                - k_yy: 纵摇回转半径与L_PP的比值
                - E_1: 水线入流角 (弧度)
                - E_2: 水线出流角 (弧度)
                - U: 船舶航速 (m/s)
                - zeta_A: 波幅 (m)
                - lambda_wave: 波长 (m)
                
        Raises:
            ValueError: 当提供的参数无效时抛出
        """
        # 默认值设置
        self.L_PP = None
        self.B = None
        self.C_B = None
        self.T_f = None
        self.T_a = None
        self.k_yy = None
        self.E_1 = None
        self.E_2 = None
        self.U = None
        self.zeta_A = None
        self.lambda_wave = None
        
        # 物理常数
        self.rho = 1025  # 海水密度 (kg/m³)
        self.g = 9.81  # 重力加速度 (m/s²)
        
        # 结果存储
        self.results = {}
        
        # 如果提供了参数，则设置参数
        if ship_params is not None:
            self.set_parameters(ship_params)

    def set_parameters(self, ship_params):
        """
        设置船舶参数并自动验证

        Args:
            ship_params (dict): 船舶参数字典，包含必要的船舶参数和计算参数
            
        Raises:
            ValueError: 当参数未设置或参数值超出有效范围时抛出
        """
        self.L_PP = ship_params['L_PP']
        self.B = ship_params['B']
        self.C_B = ship_params['C_B']
        self.T_f = ship_params['T_f']
        self.T_a = ship_params['T_a']
        self.k_yy = ship_params['k_yy']
        self.E_1 = ship_params['E_1']
        self.E_2 = ship_params['E_2']
        
        # 设置必传的计算参数
        self.U = ship_params['U']
        self.zeta_A = ship_params['zeta_A']
        self.lambda_wave = ship_params['lambda_wave']

        # 计算派生参数
        self.T_max = max(self.T_f, self.T_a)
        self.T = (self.T_f + self.T_a) / 2
        self.tau_B = self.T / self.B
        self.tau_max = self.T_max / self.B
        
        # 清空之前的计算结果
        self.results = {}
        
        # 自动验证参数
        self._validate_parameters()

    def calculate(self, omega, alpha):
        """
        计算总的波浪附加阻力

        Args:
            omega: 规则波圆频率 (rad/s)
            alpha: 船舶航向与波浪传播方向夹角 (弧度，0表示顺浪)

        Returns:
            float: 总附加阻力 R_AW (N)
            
        Raises:
            ValueError: 当必要的计算参数未设置时抛出
            TypeError: 当输入参数类型错误时抛出
        """
        # 验证输入参数类型
        if not isinstance(omega, (int, float)):
            raise TypeError("omega必须是单个数值")
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha必须是单个数值")
        
        # 计算两种阻力分量
        R_AWM = self._calculate_motion_resistance(omega, alpha)
        R_AWR = self._calculate_reflection_resistance(omega, alpha)
        R_AW_total = R_AWM + R_AWR
        
        # 存储计算结果
        self.results = {
            'R_AW_total': R_AW_total,
            'R_AWM': R_AWM,
            'R_AWR': R_AWR,
            'omega': omega,
            'alpha': alpha,
            'U': self.U,
            'zeta_A': self.zeta_A,
            'lambda_wave': self.lambda_wave
        }
        
        # 保存详细组件
        # components = self.get_motion_resistance_components(omega, alpha, U, zeta_A, lambda_wave)
        # self.results['components'] = components
        
        return R_AW_total

    def _calculate_motion_resistance(self, omega, alpha):
        """
        计算运动引起的附加阻力 R_AWM
        """


        # 计算无量纲频率
        omega_bar = self._calculate_omega_bar_corrected(omega, alpha)

        # 计算系数
        a1 = self._calculate_a1_corrected(alpha, omega)
        a2 = self._calculate_a2(alpha, omega)
        a3 = self._calculate_a3_corrected()
        b1 = self._calculate_b1(omega_bar)
        d1 = self._calculate_d1_corrected(omega_bar)

        # 计算运动附加阻力
        coefficient = 4 * self.rho * self.g * self.zeta_A ** 2 * (self.B ** 2 / self.L_PP)

        if omega_bar > 0 and d1 != 0:
            exponential_term = math.exp(b1 / d1 * (1 - omega_bar ** d1))
        else:
            exponential_term = 1.0

        R_AWM = coefficient * a1 * a2 * a3 * (omega_bar ** b1) * exponential_term

        return R_AWM

    def _calculate_omega_bar_corrected(self, omega, alpha):
        """
        计算修正的无量纲频率 ω̄
        """
        Fr = self.U / math.sqrt(self.g * self.L_PP)
        L_pp_lambda = self.L_PP / self.lambda_wave

        term1 = 2.142 * (self.k_yy ** (1 / 3)) * math.sqrt(L_pp_lambda)
        term2 = 1 - (0.111 / self.C_B) * (math.log(self.tau_B / self.tau_max) - math.log(2.75))
        term3 = (self.C_B / 0.85) ** 0.17
        term4 = (-1.377 * Fr ** 2 + 1.157 * Fr) * abs(math.cos(alpha)) + \
                0.618 * (13 + math.cos(2 * alpha)) / 14

        omega_bar = term1 * term2 * term3 * term4

        return omega_bar

    def _calculate_a1_corrected(self, alpha, omega):
        """
        计算修正的系数a1，根据分段函数公式
        
        Args:
            alpha: 航向角 (弧度)
            omega: 波浪频率 (rad/s)
            
        Returns:
            float: 修正的系数a1
        """
        V_g = self.g / (2 * omega)  # 群速度
        
        # 计算a1(0)和a1(π/2)的值
        a1_0 = self._calculate_f_U_Vg_corrected(V_g)  # 顺浪情况系数
        
        # 计算a1(π/2)的值
        Fr = self.U / math.sqrt(self.g * self.L_PP)
        a1_pi_2 = 20.1 * (self.C_B ** 1.34) * ((4 * self.k_yy) ** 2) * (math.log(self.B / self.T_max)) ** (-1)
        
        # 根据分段函数计算a1
        if alpha == 0 or alpha == 2 * math.pi:
            # 情况1: 顺浪
            a1 = a1_0
            
        elif 0 < alpha < math.pi / 2:
            # 情况2: 顺浪到横浪之间
            a1 = a1_0 + (2 * alpha / math.pi) * (a1_pi_2 - a1_0)
            
        elif math.pi / 2 <= alpha <= math.pi:
            # 情况3: 横浪到顶浪区间
            base_term = 60.3 * (self.C_B ** 1.34) * ((4 * self.k_yy) ** 2)
            power_term = (0.87 / self.C_B) ** (-(1 + Fr) * math.cos(alpha))
            log_term = (math.log(self.B / self.T_max)) ** (-1)
            cos_term = (1 - 2 * math.cos(alpha)) / 3
            a1 = base_term * power_term * log_term * cos_term
            
        elif math.pi < alpha < 3 * math.pi / 2:
            # 情况4: 顶浪到另一舷横浪区间
            phi = 2 * math.pi - alpha  # 转换为等效角度
            base_term = 60.3 * (self.C_B ** 1.34) * ((4 * self.k_yy) ** 2)
            power_term = (0.87 / self.C_B) ** (-(1 + Fr) * math.cos(phi))
            log_term = (math.log(self.B / self.T_max)) ** (-1)
            cos_term = (1 - 2 * math.cos(phi)) / 3
            a1 = base_term * power_term * log_term * cos_term
            
        elif 3 * math.pi / 2 <= alpha < 2 * math.pi:
            # 情况5: 另一舷横浪到顺浪区间
            a1 = a1_0 + (2 * (2 * math.pi - alpha) / math.pi) * (a1_pi_2 - a1_0)
            
        else:
            # 异常情况处理
            raise ValueError(f"航向角alpha = {alpha} 超出有效范围 [0, 2π]")
            
        return a1

    def _calculate_f_U_Vg_corrected(self, V_g):
        """
        计算修正的函数f(U,Vg)用于顺浪情况
        """
        # 提取公共变量计算
        base_term = 60.3 * (self.C_B ** 1.34) * ((4 * self.k_yy) ** 2)
        log_term = (math.log(self.B / self.tau_max)) ** (-1)
        power_term = (0.87 / self.C_B) ** (math.cos(math.pi))
        cos_term = (1 - 2 * math.cos(math.pi)) / 3
        full_value = base_term * power_term * log_term * cos_term

        if self.U == 0:
            return -full_value

        elif self.U == V_g / 2:
            return 0

        elif self.U == V_g:
            return full_value

        elif self.U > V_g:
            Fr_rel = (self.U - V_g) / math.sqrt(self.g * self.L_PP)
            power_term = (0.87 / self.C_B) ** (-(1 + Fr_rel) * math.cos(math.pi))
            return base_term * power_term * log_term * cos_term

        else:
            # 线性插值处理中间值
            if 0 < self.U < V_g / 2:
                a1_0 = -full_value  # U=0时的值
                a1_half = 0
                weight = self.U / (V_g / 2)
                return (1 - weight) * a1_0 + weight * a1_half
            elif V_g / 2 < self.U < V_g:
                a1_half = 0
                a1_full = full_value  # U=Vg时的值
                weight = (self.U - V_g / 2) / (V_g / 2)
                return (1 - weight) * a1_half + weight * a1_full

        return 0

    def _calculate_a2(self, alpha, omega):
        """
        根据分段函数计算系数a2(α)
        覆盖0 ≤ α < 2π的完整区间
        """   

        # 计算Froude数
        Fr = self.U / math.sqrt(self.g * self.L_PP)
        # 计算a2_formula（用于横浪和顶浪区间）
        if Fr < 0.12:
            a2_formula = 0.0072 + 0.1676 * Fr
        else:
            a2_formula = (Fr ** 1.5) * math.exp(-3.5 * Fr)
        
        # 设置a2(0)和a2(π/2)的值
        a2_0 = self.calculate_a2_0(omega)
        a2_pi_2 = a2_formula  # 横浪时a2的值等于a2_formula
        
        # 根据不同的角度区间计算a2(α)
        if alpha == 0 or alpha == 2 * math.pi:
            # 情况1: 顺浪 (α=0 或 α=2π)
            return a2_0
            
        elif 0 < alpha < math.pi / 2:
            # 情况2: 顺浪到横浪区间
            return a2_0 + (2 * alpha / math.pi) * (a2_pi_2 - a2_0)
            
        elif math.pi / 2 <= alpha <= math.pi:
            # 情况3: 横浪到顶浪区间
            return a2_formula
            
        elif math.pi < alpha < 3 * math.pi / 2:
            # 情况4: 顶浪到另一舷横浪区间
            return a2_formula
            
        elif 3 * math.pi / 2 <= alpha < 2 * math.pi:
            # 情况5: 另一舷横浪到顺浪区间
            return a2_0 + (2 * (2 * math.pi - alpha) / math.pi) * (a2_pi_2 - a2_0)
        
        # 默认返回值（防止异常情况）
        return 0

    # 计算a2(0)（顺浪值）
    def calculate_a2_0(self,omega):
        # 计算波浪群速度
        V_g = self.g / (2 * omega)
        if self.U == 0:
            return -0.0072
        elif self.U == V_g / 2:
            return 0
        elif self.U == V_g:
            return 0.0072
        elif self.U > V_g:
            Fr_rel = (self.U - V_g) / math.sqrt(self.g * self.L_PP)
            if Fr_rel < 0.12:
                return 0.0072 + 0.1676 * Fr_rel
            else:
                return (Fr_rel ** 1.5) * math.exp(-3.5 * Fr_rel)
        else:
            # 线性插值处理中间值
            if 0 < self.U < V_g / 2:
                a2_0_val = -0.0072
                a2_half_val = 0
                weight = self.U / (V_g / 2)
                return (1 - weight) * a2_0_val + weight * a2_half_val
            elif V_g / 2 < self.U < V_g:
                a2_half_val = 0
                a2_full_val = 0.0072
                weight = (self.U - V_g / 2) / (V_g / 2)
                return (1 - weight) * a2_half_val + weight * a2_full_val
        return 0

    def _calculate_a3_corrected(self):
        """
        计算修正的系数a3
        """
        draft_diff = abs(self.T_a - self.T_f)
        atan_term = math.atan(draft_diff / self.L_PP)

        a3 = 1.0 + 28.7 * atan_term

        return a3

    def _calculate_b1(self, omega_bar):
        """
        计算系数b1
        """
        return 11.0 if omega_bar < 1 else -8.5

    def _calculate_d1_corrected(self, omega_bar):
        """
        计算修正的系数d1
        """
        base_term = 566 * ((self.L_PP * self.C_B / self.B) ** (-2.66))

        if omega_bar < 1:
            return base_term
        else:
            atan_term = math.atan(abs(self.T_a - self.T_f) / self.L_PP)
            return -base_term * (4 - 125 * atan_term)

    def _calculate_reflection_resistance(self, omega, alpha):
        """
        计算反射引起的附加阻力 R_AWR
        """
        R_AWR_total = 0

        # 计算四个分量
        R_AWR_total += self._calculate_R_AWR_1(omega, alpha)
        R_AWR_total += self._calculate_R_AWR_2(omega, alpha)
        R_AWR_total += self._calculate_R_AWR_3(omega, alpha)
        R_AWR_total += self._calculate_R_AWR_4(omega, alpha)

        return R_AWR_total

    def _calculate_R_AWR_1(self, omega, alpha):
        """计算R_AWR第一项"""
        if not (self.E_1 <= alpha <= math.pi):
            return 0

        Fr = self.U / math.sqrt(self.g * self.L_PP)

        # 计算吃水系数
        alpha_T_star = self._calculate_alpha_T_star_corrected(self.lambda_wave, self.T_max)

        # 计算基础项
        base_coefficient = (2.25 / 4) * self.rho * self.g * self.B * (self.zeta_A ** 2) * alpha_T_star

        # 计算角度相关项
        angle_term = (math.sin(self.E_1 - alpha) ** 2 +
                      (2 * omega / self.g) *
                      (math.cos(self.E_1) * math.cos(self.E_1 - alpha) - math.cos(alpha)))

        # 计算修正项
        correction_term = (0.87 / self.C_B) ** (1 + 4 * math.sqrt(Fr))

        # 计算f(α)
        f_alpha = self._calculate_f_alpha(alpha)

        return base_coefficient * angle_term * correction_term * f_alpha

    def _calculate_R_AWR_2(self, omega, alpha):
        """计算R_AWR第二项"""
        if not (math.pi - self.E_1 <= alpha <= math.pi):
            return 0

        Fr = self.U / math.sqrt(self.g * self.L_PP)

        alpha_T_star = self._calculate_alpha_T_star_corrected(self.lambda_wave, self.T_max)

        base_coefficient = (2.25 / 4) * self.rho * self.g * self.B * (self.zeta_A ** 2) * alpha_T_star

        angle_term = (math.sin(self.E_1 + alpha) ** 2 +
                      (2 * omega / self.g) *
                      (math.cos(self.E_1) * math.cos(self.E_1 + alpha) - math.cos(alpha)))

        correction_term = (0.87 / self.C_B) ** (1 + 4 * math.sqrt(Fr))
        f_alpha = self._calculate_f_alpha(alpha)

        return base_coefficient * angle_term * correction_term * f_alpha

    def _calculate_R_AWR_3(self, omega, alpha):
        """计算R_AWR第三项"""
        if not (0 <= alpha <= math.pi - self.E_2):
            return 0

        # 计算T*
        T_star = self._calculate_T_star_for_3_4(alpha)
        alpha_T_star = self._calculate_alpha_T_star_corrected(self.lambda_wave, T_star)

        base_coefficient = -(2.25 / 4) * self.rho * self.g * self.B * (self.zeta_A ** 2) * alpha_T_star

        angle_term = (math.sin(self.E_2 + alpha) ** 2 +
                      (2 * omega / self.g) *
                      (math.cos(self.E_2) * math.cos(self.E_2 + alpha) - math.cos(alpha)))

        return base_coefficient * angle_term

    def _calculate_R_AWR_4(self, omega, alpha):
        """计算R_AWR第四项"""
        if not (0 <= alpha <= self.E_2):
            return 0

        T_star = self._calculate_T_star_for_3_4(alpha)
        alpha_T_star = self._calculate_alpha_T_star_corrected(self.lambda_wave, T_star)

        base_coefficient = -(2.25 / 4) * self.rho * self.g * self.B * (self.zeta_A ** 2) * alpha_T_star

        angle_term = (math.sin(self.E_2 - alpha) ** 2 +
                      (2 * omega / self.g) *
                      (math.cos(self.E_2) * math.cos(self.E_2 - alpha) - math.cos(alpha)))

        return base_coefficient * angle_term

    def _calculate_alpha_T_star_corrected(self, lambda_wave, T_star):
        """
        计算修正的吃水系数α_T*
        """
        lambda_L_ratio = lambda_wave / self.L_PP

        if lambda_L_ratio <= 2.5:
            term = -4 * math.pi * (T_star / lambda_wave - T_star / (2.5 * self.L_PP))
            return 1 - math.exp(term)
        else:
            return 0

    def _calculate_T_star_for_3_4(self, alpha):
        """
        计算R_AWR_3和R_AWR_4项的修正吃水T*
        """
        if self.C_B <= 0.75:
            return self.T_max * (4 + math.sqrt(abs(math.cos(alpha)))) / 5
        else:
            return self.T_max * (2 + math.sqrt(abs(math.cos(alpha)))) / 3

    def _calculate_f_alpha(self, alpha):
        """
        计算函数f(α)
        """
        if math.pi - self.E_1 <= alpha < math.pi:
            return -math.cos(alpha)
        else:
            return 0

    def _validate_parameters(self):
        """
        内部方法：验证船舶参数是否在有效范围内

        Raises:
            ValueError: 当参数未设置或参数值超出有效范围时抛出
        """
        # 检查船舶参数是否已设置
        if self.L_PP is None:
            raise ValueError("船舶参数 L_PP 未设置")
        if self.B is None:
            raise ValueError("船舶参数 B 未设置")
        if self.C_B is None:
            raise ValueError("船舶参数 C_B 未设置")
        if self.T_f is None:
            raise ValueError("船舶参数 T_f 未设置")
        if self.T_a is None:
            raise ValueError("船舶参数 T_a 未设置")
        if self.k_yy is None:
            raise ValueError("船舶参数 k_yy 未设置")
        if self.E_1 is None:
            raise ValueError("船舶参数 E_1 未设置")
        if self.E_2 is None:
            raise ValueError("船舶参数 E_2 未设置")
        if self.U is None:
            raise ValueError("计算参数 U 未设置")
        if self.zeta_A is None:
            raise ValueError("计算参数 zeta_A 未设置")
        if self.lambda_wave is None:
            raise ValueError("计算参数 lambda_wave 未设置")
        
        # 检查计算参数的物理合理性
        if self.U < 0:
            raise ValueError("船速U不能为负值")
        if self.zeta_A <= 0:
            raise ValueError("波幅zeta_A必须为正数")
        if self.lambda_wave <= 0:
            raise ValueError("波长lambda_wave必须为正数")
        
        # 检查参数范围
        if not (80 <= self.L_PP <= 380):
            raise ValueError(f"L_PP = {self.L_PP} 不在范围 [80, 380] m")

        if not (0.54 <= self.C_B <= 0.87):
            raise ValueError(f"C_B = {self.C_B} 不在范围 [0.54, 0.87]")

        # 检查物理合理性
        if self.tau_B <= 0 or self.tau_max <= 0:
            raise ValueError("吃水比值计算错误")

        if self.k_yy <= 0:
            raise ValueError("纵摇回转半径比必须为正数")

    def get_motion_resistance_components(self, omega, alpha):
        """
        获取运动附加阻力的各个组成部分

        Args:
            omega: 波浪频率 (rad/s)
            alpha: 航向角 (弧度)

        Returns:
            dict: 包含各系数和分量的字典
        """
        omega_bar = self._calculate_omega_bar_corrected(omega, alpha)
        a1 = self._calculate_a1_corrected(alpha, omega)
        a2 = self._calculate_a2(alpha, omega)
        a3 = self._calculate_a3_corrected()
        b1 = self._calculate_b1(omega_bar)
        d1 = self._calculate_d1_corrected(omega_bar)

        coefficient = 4 * self.rho * self.g * self.zeta_A ** 2 * (self.B ** 2 / self.L_PP)

        if omega_bar > 0 and d1 != 0:
            exponential_term = math.exp(b1 / d1 * (1 - omega_bar ** d1))
        else:
            exponential_term = 1.0

        R_AWM = coefficient * a1 * a2 * a3 * (omega_bar ** b1) * exponential_term

        return {
            'Fr': Fr,
            'omega_bar': omega_bar,
            'a1': a1,
            'a2': a2,
            'a3': a3,
            'b1': b1,
            'd1': d1,
            'coefficient': coefficient,
            'exponential_term': exponential_term,
            'R_AWM': R_AWM
        }
        
    def get_results(self):
        """
        获取最近一次计算的结果
        
        Returns:
            dict: 包含计算结果的字典
            
        Raises:
            ValueError: 当尚未进行计算时抛出
        """
        if not self.results:
            raise ValueError("尚未进行计算，请先调用calculate方法")
        
        return self.results.copy()

# 使用示例
if __name__ == "__main__":
    # 示例1：使用构造函数初始化所有参数并计算
    try:
        ship_params = {
            'L_PP': 175,  # 垂线间长 (m)
            'B': 25,  # 船宽 (m)
            'C_B': 0.75,  # 方形系数
            'T_f': 8.5,  # 前垂线吃水 (m)
            'T_a': 9.0,  # 后垂线吃水 (m)
            'k_yy': 0.25,  # 纵摇回转半径比
            'E_1': 0.52,  # 入流角 (弧度，约30度)
            'E_2': 0.35,  # 出流角 (弧度，约20度)
            'U': 10,  # 船速 (m/s)
            'zeta_A': 2,  # 波幅 (m)
            'lambda_wave': 150  # 波长 (m)
        }
        
        calculator = WaveAddedResistanceCalculator(ship_params)
        omega = 0.5  # 波浪频率 (rad/s)
        alpha = math.pi  # 顶浪 (弧度)
        
        R_AW = calculator.calculate(omega, alpha)
        print(f"示例1 - 顶浪情况下的波浪附加阻力: {R_AW:.2f} N")
        
        # 示例2：使用set_parameters方法设置所有参数
        calculator2 = WaveAddedResistanceCalculator()
        ship_params2 = {
            'L_PP': 200,
            'B': 30,
            'C_B': 0.80,
            'T_f': 9.0,
            'T_a': 9.5,
            'k_yy': 0.27,
            'E_1': 0.61,
            'E_2': 0.44,
            'U': 12,
            'zeta_A': 1.5,
            'lambda_wave': 180
        }
        calculator2.set_parameters(ship_params2)
        
        R_AW2 = calculator2.calculate(omega, math.pi/2)  # 横浪情况
        print(f"示例2 - 横浪情况下的波浪附加阻力: {R_AW2:.2f} N")
        
        # 示例3：顺浪情况下的计算
        R_AW3 = calculator.calculate(omega, 0)  # 顺浪
        print(f"示例3 - 顺浪情况下的波浪附加阻力: {R_AW3:.2f} N")
        
        # 示例5：获取详细计算结果
        results = calculator.get_results()
        print(f"\n示例5 - 详细计算结果:")
        print(f"总阻力: {results['R_AW_total']:.2f} N")
        print(f"运动阻力分量: {results['R_AWM']:.2f} N")
        print(f"反射阻力分量: {results['R_AWR']:.2f} N")
        print(f"计算参数 - 船速: {results['U']} m/s, 波幅: {results['zeta_A']} m, 波长: {results['lambda_wave']} m")
    except ValueError as e:
        print(f"参数验证错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")