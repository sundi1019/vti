#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
温度相关的流体动力学计算模块
实现雷诺数、摩擦阻力系数和海水密度的计算
"""

import math
import numpy as np
from K import K_FactorCalculator


class FluidDynamicsCalculator:
    """
    流体动力学计算类，统一管理所有参数和计算方法
    """
    
    def __init__(self, ship_params=None):
        """
        初始化流体动力学计算器
        
        Args:
            ship_params (dict, optional): 参数字典，包含：
                - velocity: 对水速度 (m/s)
                - Lpp: 垂线间长 (m)
                - temperature: 水温 (°C)，默认为15°C
                - L_wl: 水线长 (m)，用于所有计算
                - kS: 船体表面粗糙度 (m)，默认为150×10^-6 m
                - wet_surface_area: 船舶湿表面积 (m²)
                - Pds: 实测交付功率 (kW)
                - Rawirge: 波浪附加阻力 (N)
                - Rwind: 风阻力 (N)
                - nt: 船舶总效率
                - B: 型宽 (m)，用于计算形状因子k
                - T: 型深 (m)，用于计算形状因子k
                - Delta: 型排水量 (m³)，用于计算形状因子k
                - A_m: 中剖面面积 (m²)，用于计算形状因子k
                - aft_body_shape: 尾部形状，用于计算形状因子k
        """
        # 初始化默认参数
        self.velocity = None
        self.Lpp = None
        self.temperature = 15.0  # 默认温度
        self.L_wl = None
        self.kS = 150e-6  # 默认表面粗糙度
        self.wet_surface_area = None
        self.Pds = None
        self.Rawirge = None
        self.Rwind = None
        self.nt = None
        # 形状因子k计算所需的船舶参数
        self.B = None
        self.T = None
        self.Delta = None
        self.A_m = None
        self.aft_body_shape = None
        
        # 结果存储
        self.results = {}
        
        # 如果提供了参数，则设置参数
        if ship_params is not None:
            self.set_parameters(ship_params)
    
    def set_parameters(self, ship_params):
        """
        设置计算所需的参数
        
        Args:
            ship_params (dict): 参数字典，包含流体动力学计算所需的参数
        """
        # 设置流体动力学计算参数
        self.velocity = ship_params.get('velocity', self.velocity)  # 对水速度 (m/s)
        self.Lpp = ship_params.get('Lpp', ship_params.get('length_ref', self.Lpp))  # 支持Lpp和length_ref两种参数名
        self.temperature = ship_params.get('temperature', self.temperature)
        # 优先使用L_wl，如果没有则尝试L_WL，保持向后兼容
        self.L_wl = ship_params.get('L_wl', ship_params.get('L_WL', self.L_wl))
        self.kS = ship_params.get('kS', self.kS)
        self.wet_surface_area = ship_params.get('wet_surface_area', self.wet_surface_area)
        self.Pds = ship_params.get('Pds', self.Pds)
        self.Rawirge = ship_params.get('Rawirge', self.Rawirge)
        self.Rwind = ship_params.get('Rwind', self.Rwind)
        self.nt = ship_params.get('nt', self.nt)
        # 设置形状因子k计算所需的船舶参数
        self.B = ship_params.get('B', self.B)
        self.T = ship_params.get('T', self.T)
        self.Delta = ship_params.get('Delta', self.Delta)
        self.A_m = ship_params.get('A_m', self.A_m)
        self.aft_body_shape = ship_params.get('aft_body_shape', self.aft_body_shape)
        
        # 清空之前的计算结果
        self.results = {}
        
        # 统一校验所有输入参数
        self._validate_parameters()
    
    def calculate(self):
        """
        主要计算入口方法，返回温度修正的总阻力Rtemp
        
        Returns:
            float: 温度修正的总阻力Rtemp (N)
        """
        
        # 调用calculate_Rtemp方法计算温度修正的总阻力
        return self.calculate_Rtemp()
    
    def _validate_parameters(self):
        """
        统一校验所有输入参数
        """
        # 速度验证
        if self.velocity < 0:
            raise ValueError("对水速度不能为负值")
        
        # 垂线间长验证
        if self.Lpp <= 0:
            raise ValueError("垂线间长必须大于0")
        
        # 温度验证
        if self.temperature < -2 or self.temperature > 40:
            raise ValueError("水温应在合理范围内(-2°C到40°C)")
        
        # 表面粗糙度验证
        if self.kS < 0:
            raise ValueError("表面粗糙度不能为负值")
        
        # 湿表面积验证
        if self.wet_surface_area is not None and self.wet_surface_area <= 0:
            raise ValueError("湿表面积必须大于0")
        
        # 功率验证（如果提供）
        if self.Pds is not None and self.Pds < 0:
            raise ValueError("功率不能为负值")
        
        # 效率验证（如果提供）
        if self.nt is not None and (self.nt <= 0 or self.nt > 1):
            raise ValueError("效率应在0到1之间")
        
        # 阻力验证（如果提供）
        if self.Rawirge is not None and self.Rawirge < 0:
            raise ValueError("波浪附加阻力不能为负值")
        
        if self.Rwind is not None and self.Rwind < 0:
            raise ValueError("风阻力不能为负值")
        
        # 形状因子k相关参数验证（如果提供）
        if self.B is not None and self.B <= 0:
            raise ValueError("型宽B必须大于0")
        if self.T is not None and self.T <= 0:
            raise ValueError("型深T必须大于0")
        if self.Delta is not None and self.Delta <= 0:
            raise ValueError("型排水量Delta必须大于0")
        if self.A_m is not None and self.A_m <= 0:
            raise ValueError("中剖面面积A_m必须大于0")
        
        # 垂线间长验证（继续下面的代码）
        if self.Lpp <= 0:
            raise ValueError("垂线间长必须大于0")
        
        # 温度验证
        if self.temperature < 0 or self.temperature > 100:
            raise ValueError(f"温度 {self.temperature}°C 超出合理范围 (0-100°C)")
        
        # 水线长验证（如果提供）
        if self.L_wl is not None and self.L_wl <= 0:
            raise ValueError("水线长必须大于0")
        
        # 表面粗糙度验证
        if self.kS <= 0:
            raise ValueError("表面粗糙度必须大于0")
        
        # 湿表面积验证（如果提供）
        if self.wet_surface_area is not None and self.wet_surface_area <= 0:
            raise ValueError("船舶湿表面积必须大于0")
    
    def calculate_kinematic_viscosity(self, temperature=None):
        """
        计算水的运动粘度
        
        根据公式: μ(t) = 1.83×10^-6 - 5.23×10^-8 × t + 6.59×10^-10 × t²
        
        Args:
            temperature (float, optional): 水温 (°C)，默认为None（使用实例的温度值）
            
        Returns:
            float: 水的运动粘度 (m²/s)
        """
        # 使用传入的温度或默认使用实例温度
        temp_to_use = temperature if temperature is not None else self.temperature
        
        # 运动粘度计算
        mu = 1.83e-6 - 5.23e-8 * temp_to_use + 6.59e-10 * temp_to_use**2
        return mu
    
    def calculate_reynolds_number(self, temperature=None):
        """
        计算雷诺数
        
        根据公式: Re = (v × L_ref) / μ
        
        Args:
            temperature (float, optional): 水温 (°C)，默认为None（使用实例的温度值）
            
        Returns:
            float: 雷诺数
        """
        # 计算运动粘度，传递温度参数
        mu = self.calculate_kinematic_viscosity(temperature=temperature)
        
        # 计算雷诺数
        reynolds_number = (self.velocity * self.Lpp) / mu
        return reynolds_number
    
    def calculate_Cf(self, temperature=None):
        """
        计算摩擦阻力系数 (Cf)
        
        根据公式: Cf = 0.075 / (log10(Re) - 2)^2
        
        Args:
            temperature (float, optional): 计算时使用的水温 (°C)，默认为None（使用实例的温度值）
            
        Returns:
            float: 摩擦阻力系数
        """
        # 获取雷诺数，传递温度参数
        reynolds_number = self.calculate_reynolds_number(temperature=temperature)
        
        # 参数验证
        if reynolds_number < 1000:
            raise ValueError(f"雷诺数 {reynolds_number} 过低，公式不适用于层流流动")
        
        # 计算摩擦阻力系数
        log_re = math.log10(reynolds_number)
        friction_coefficient = 0.075 / ((log_re - 2.0) ** 2)
        
        return friction_coefficient
    
    def calculate_ΔCF(self, temperature=None):
        """
        计算粗糙度修正系数 ΔCF
        
        根据公式: ΔCF = 0.044((kS/LWL)^(1/3) - 10·Re^(-1/3)) + 0.000125
        
        Args:
            temperature (float, optional): 计算时使用的水温 (°C)，默认为None（使用实例的温度值）
            
        Returns:
            float: 粗糙度修正系数 ΔCF
        """
        # 确保提供了水线长
        if self.L_wl is None:
            raise ValueError("计算粗糙度修正系数需要提供水线长 L_wl")
        
        # 获取雷诺数，传递温度参数
        reynolds_number = self.calculate_reynolds_number(temperature=temperature)
        
        # 计算粗糙度修正系数
        term1 = (self.kS / self.L_wl) ** (1/3)
        term2 = 10 * (reynolds_number ** (-1/3))
        delta_CF = 0.044 * (term1 - term2) + 0.000125
        
        # 确保修正系数非负
        delta_CF = max(0, delta_CF)
        
        return delta_CF
    
    def calculate_seawater_density(self, temperature=None):
        """
        计算海水密度
        
        根据公式: ρs(T) = 1028.2 - 0.0787·T - 0.0045·T²
        
        Args:
            temperature (float, optional): 水温 (°C)，默认为None（使用实例的温度值）
            
        Returns:
            float: 海水密度 (kg/m³)
        """
        # 使用传入的温度或默认使用实例温度
        temp_to_use = temperature if temperature is not None else self.temperature
        
        # 计算海水密度
        density = 1028.2 - 0.0787 * temp_to_use - 0.0045 * temp_to_use**2
        return density
    
    def calculate_RF(self):
        """
        计算船舶的基本摩擦阻力
        
        根据公式: RF = CF × ½ × ρs × V² × S
            
        Returns:
            float: 基本摩擦阻力 (N)
        """
        # 确保提供了湿表面积
        if self.wet_surface_area is None:
            raise ValueError("计算摩擦阻力需要提供船舶湿表面积")
        
        # 计算摩擦阻力系数（使用指定温度或默认温度）
        cf = self.calculate_Cf()
        
        # 计算海水密度，传递相同的温度参数
        seawater_density = self.calculate_seawater_density()
        
        # 计算基本摩擦阻力
        rf_basic = cf * 0.5 * seawater_density * self.velocity**2 * self.wet_surface_area
        
        return rf_basic
        
    def calculate_Rtemp(self):
        """
        计算温度修正的总阻力Rtemp
        
        根据公式: Rtemp = RT0 × (ρS/ρS0 - 1) - RF × ((CF0+ΔCF0)/(CF+ΔCF) - 1)
        
        其中:
        - RT0: 在参考水温下的总阻力 (N)，通过calculate_RT0方法计算
        - ρS: 在实际水温下的海水密度
        - ρS0: 在参考温度15°C时的海水密度 = 1026 kg/m³
        - RF: 在实际水温和实际密度下的摩擦阻力 (N)
        - CF0: 在参考水温和参考密度下的摩擦阻力系数
        - ΔCF0: 与参考水温和盐度对应的雷诺数相关的粗糙度补贴
        - CF: 在实际水温和实际密度下的摩擦阻力系数
        - ΔCF: 与实际水温和盐度对应的雷诺数相关的粗糙度补贴
        
        Returns:
            float: 温度修正的总阻力Rtemp (N)
            
        Raises:
            ValueError: 当缺少计算所需的参数时抛出
        """
        print("\n=== 开始计算温度修正的总阻力Rtemp ===")
        
        # 打印输入参数
        print(f"\n1. 输入参数:")
        print(f"   - 当前温度: {self.temperature:.2f}°C")
        print(f"   - 速度: {self.velocity:.2f} m/s")
        print(f"   - 垂线间长: {self.Lpp:.2f} m")
        print(f"   - 湿表面积: {self.wet_surface_area:.2f} m²")
        print(f"   - 实测交付功率(Pds): {self.Pds:.2f} kW")
        print(f"   - 波浪附加阻力(Rawirge): {self.Rawirge:.2f} N")
        print(f"   - 风阻力(Rwind): {self.Rwind:.2f} N")
        print(f"   - 船舶总效率(nt): {self.nt:.3f}")
        
        # 打印形状因子k相关参数
        print(f"\n2. 形状因子k计算参数:")
        print(f"   - 水线长(L_wl): {self.L_wl:.2f} m")
        print(f"   - 型宽(B): {self.B:.2f} m")
        print(f"   - 型深(T): {self.T:.2f} m")
        print(f"   - 型排水量(Delta): {self.Delta:.2f} m³")
        print(f"   - 中剖面面积(A_m): {self.A_m:.2f} m²")
        print(f"   - 尾部形状(aft_body_shape): {self.aft_body_shape}")
        
        # 计算形状因子k
        k_calculator = K_FactorCalculator()
        # 构建船舶参数字典
        ship_params = {
            'L_WL': self.L_wl,  # 传递统一的L_wl参数
            'B': self.B,
            'T': self.T,
            'Delta': self.Delta,
            'A_m': self.A_m,
            'aft_body_shape': self.aft_body_shape,
            'V_W': self.velocity  # 使用当前对象的速度
        }
        k_calculator.set_parameters(ship_params)
        k = k_calculator.calculate()
        print(f"   - 计算得到的形状因子k: {k:.6f}")
        
        # 计算RT0
        print(f"\n3. 计算参考水温下的总阻力RT0:")
        RT0 = self.calculate_RT0(self.Pds, self.Rawirge, self.Rwind, self.nt, k)
        print(f"   - RT0 = {RT0:.2f} N")
        
        # 获取当前温度下的参数
        print(f"\n4. 获取当前温度({self.temperature}°C)下的参数:")
        rho_S = self.calculate_seawater_density(temperature=None)  # 实际水温下的海水密度
        print(f"   - 实际水温下的海水密度(ρS): {rho_S:.2f} kg/m³")
        
        RF = self.calculate_RF()  # 实际水温下的摩擦阻力
        print(f"   - 实际水温下的摩擦阻力(RF): {RF:.2f} N")
        
        CF = self.calculate_Cf(temperature=None)  # 实际水温下的摩擦阻力系数
        print(f"   - 实际水温下的摩擦阻力系数(CF): {CF:.8f}")
        
        delta_CF = self.calculate_ΔCF(temperature=None)  # 实际水温下的粗糙度修正系数
        print(f"   - 实际水温下的粗糙度修正系数(ΔCF): {delta_CF:.8f}")
        print(f"   - CF + ΔCF = {CF + delta_CF:.8f}")
        
        # 获取15°C（参考温度）下的参数
        print(f"\n5. 获取参考温度(15°C)下的参数:")
        rho_S0 = 1026.0  # 在参考温度15°C时的海水密度 = 1026 kg/m³
        print(f"   - 参考温度下的海水密度(ρS0): {rho_S0:.2f} kg/m³")
        print(f"   - ρS/ρS0 - 1 = {rho_S / rho_S0 - 1:.8f}")
        
        CF0 = self.calculate_Cf(temperature=15.0)  # 参考水温下的摩擦阻力系数
        print(f"   - 参考水温下的摩擦阻力系数(CF0): {CF0:.8f}")
        
        delta_CF0 = self.calculate_ΔCF(temperature=15.0)  # 参考水温下的粗糙度修正系数
        print(f"   - 参考水温下的粗糙度修正系数(ΔCF0): {delta_CF0:.8f}")
        print(f"   - CF0 + ΔCF0 = {CF0 + delta_CF0:.8f}")
        print(f"   - (CF0 + ΔCF0)/(CF + ΔCF) - 1 = {(CF0 + delta_CF0)/(CF + delta_CF) - 1:.8f}")
        
        # 计算温度修正的总阻力Rtemp
        print(f"\n6. 计算温度修正的总阻力Rtemp:")
        term1 = RT0 * (rho_S / rho_S0 - 1)
        print(f"   - 第一项: RT0 × (ρS/ρS0 - 1) = {RT0:.2f} × {rho_S / rho_S0 - 1:.8f} = {term1:.2f} N")
        
        term2 = RF * ((CF0 + delta_CF0) / (CF + delta_CF) - 1)
        print(f"   - 第二项: RF × ((CF0+ΔCF0)/(CF+ΔCF) - 1) = {RF:.2f} × {(CF0 + delta_CF0)/(CF + delta_CF) - 1:.8f} = {term2:.2f} N")
        
        Rtemp = term1 - term2
        print(f"   - Rtemp = 第一项 - 第二项 = {term1:.2f} - {term2:.2f} = {Rtemp:.2f} N")
        
        print(f"\n=== Rtemp计算完成 ===")
        
        return Rtemp
        
    def calculate_RT0(self, Pds, Rawirge, Rwind, nt, k):
        """
        计算总阻力RT0
        
        根据公式组合：
        1. RT = (P_DS - P_wave - P_wind) * η_t / V_W
           其中 P_wave = Rawirge * V_w / η_t，P_wind = Rwind * V_w / η_t
           简化后：RT = P_DS * η_t / V_W - Rawirge - Rwind
        2. CW = RT / factor - (1 + k) * CF，其中factor = (1/2) × ρs0 × VW² × S（ρs0为15°C时的海水密度）
        3. CT0 = CF0(1+k) + CW，其中CF0为15°C时的摩擦阻力系数，CF为当前温度的摩擦阻力系数
        4. RT0 = CT0 × (1/2) × ρs0 × VW² × S（ρs0为15°C时的海水密度）
        其中 S 默认为 self.wet_surface_area
        
        Args:
            Pds (float): 实测交付功率 (kW)
            Rawirge (float): 波浪附加阻力 (N)
            Rwind (float): 风阻力 (N)
            nt (float): 船舶总效率
            k (float): 形状因子
            
        Returns:
            float: 总阻力RT0 (N)
        """
        # 参数验证
        if nt <= 0:
            raise ValueError("船舶总效率必须大于0")
        
        if self.velocity <= 0:
            raise ValueError("速度必须大于0")
        
        if self.wet_surface_area is None:
            raise ValueError("计算RT0需要提供船舶湿表面积")
        
        # 不需要S0参数
        
        # 将功率从kW转换为W
        Pds_watts = Pds * 1000
        
        # 根据公式计算RT = (P_DS - P_wave - P_wind) * η_t / V_W
        # 其中 P_wave = Rawirge * V_w / η_t，P_wind = Rwind * V_w / η_t
        # 代入后得到：RT = (Pds_watts - Rawirge*self.velocity/nt - Rwind*self.velocity/nt) * nt / self.velocity
        # 简化后：RT = Pds_watts * nt / self.velocity - Rawirge - Rwind
        RT = (Pds_watts * nt / self.velocity) - Rawirge - Rwind
        
        # 计算CF0（T=15°C时的摩擦阻力系数）
        CF0 = self.calculate_Cf(temperature=15.0)
        
        # 计算CF（当前温度下的摩擦阻力系数）
        CF = self.calculate_Cf(temperature=None)  # 使用实例的温度值
        
        # 计算15°C时的海水密度(ρs0)用于计算factor和CW
        rho_s0 = self.calculate_seawater_density(temperature=15.0)
        
        # 计算factor = (1/2) × ρs0 × VW² × S（根据公式移除S0）
        factor = 0.5 * rho_s0 * self.velocity**2 * self.wet_surface_area
        
        # 计算CW
        CW = RT / factor - (1 + k) * CF
        
        # 计算CT0
        CT0 = CF0 * (1 + k) + CW
        
        # 计算RT0 = CT0 × (1/2) × ρs0 × VW² × S
        RT0 = CT0 * factor
        
        # 确保结果非负
        RT0 = max(0, RT0)
        
        return RT0
    
    def calculate_combined(self):
        """
        组合计算所有可能的结果
        
        Returns:
            dict: 包含计算结果的字典
        """
        # 计算基本参数（传递None使用实例温度）
        mu = self.calculate_kinematic_viscosity(temperature=None)
        re = self.calculate_reynolds_number(temperature=None)
        cf = self.calculate_Cf(temperature=None)  # 使用实例温度
        cf_15 = self.calculate_Cf(temperature=15.0)  # T=15°C时的摩擦阻力系数
        seawater_density = self.calculate_seawater_density(temperature=None)  # 使用实例温度
        
        # 构建结果字典
        results = {
            'velocity': self.velocity,  # 对水速度
            'Lpp': self.Lpp,  # 垂线间长
            'temperature': self.temperature,
            'kinematic_viscosity': mu,
            'reynolds_number': re,
            'friction_coefficient': cf,
            'friction_coefficient_15C': cf_15,
            'seawater_density': seawater_density
        }
        
        # 如果提供了水线长，则计算粗糙度修正
        if self.L_wl is not None:
            delta_CF = self.calculate_ΔCF()
            results.update({
                'L_wl': self.L_wl,
                'kS': self.kS,
                'roughness_correction': delta_CF
            })
        
        # 如果提供了湿表面积，则计算摩擦阻力
        if self.wet_surface_area is not None:
            rf_basic = self.calculate_RF()
            results.update({
                'RF': rf_basic,
                'wet_surface_area': self.wet_surface_area
            })
        
        return results


# 注：已移除所有向后兼容的函数接口，仅保留FluidDynamicsCalculator类接口


if __name__ == "__main__":
    print("=== 流体动力学计算示例 ===")
    
    # 示例：计算温度修正的总阻力Rtemp
    print("\n2. 计算温度修正的总阻力Rtemp示例:")
    try:
        # 创建计算器实例
        rtemp_calculator = FluidDynamicsCalculator()
        
        # 设置参数
        ship_params = {
            'velocity': 6.17,
            'Lpp': 100.0,
            'temperature': 20.0,
            'L_wl': 110.0,
            'kS': 150e-6,
            'wet_surface_area': 7869.29,
            'Pds': 5000.0,         # 实测交付功率 (kW)
            'Rawirge': 13267.06,     # 波浪附加阻力 (N)
            'Rwind': 5474.67,       # 风阻力 (N)
            'nt': 0.827,             # 船舶总效率
            'L_WL': 110.0,         # 水线长 (m)
            'B': 15.0,             # 型宽 (m)
            'T': 5.0,              # 型深 (m)
            'Delta': 5000.0,       # 型排水量 (m³)
            'A_m': 75.0,           # 中剖面面积 (m²)
            'aft_body_shape': 'normal sections'  # 尾部形状
        }
        rtemp_calculator.set_parameters(ship_params)
        
        # 计算温度修正的总阻力Rtemp
        rtemp = rtemp_calculator.calculate()
        print(f"温度修正的总阻力Rtemp: {rtemp:.2f} N")
        
    except ValueError as e:
        print(f"计算Rtemp时出错: {e}")
    except ImportError:
        print("计算Rtemp时出错: 缺少必要的依赖模块 vti.K")