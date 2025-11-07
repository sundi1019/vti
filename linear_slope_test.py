#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
船舶稳态检测 - 线性斜率检验方法

基于DNV AS技术规则实现的稳态检测方法B（线性斜率检验）
该方法通过检验时间序列数据的线性趋势是否显著来判断数据是否处于稳态。
"""

import numpy as np
from scipy import stats
from typing import Union, List, Optional, Dict


class LinearSlopeTest:
    """
    线性斜率检验类
    
    实现DNV AS中描述的稳态检测方法B，用于判断船舶性能数据是否处于稳态。
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        初始化线性斜率检验器
        
        参数:
            alpha (float): 显著性水平，默认0.01（99%置信水平）
        """
        if not 0 < alpha < 1:
            raise ValueError("显著性水平alpha必须在0到1之间")
        self.alpha = alpha
    
    def _calculate_slope(self, data: np.ndarray, t: np.ndarray) -> float:
        """
        使用最小二乘法计算斜率b₁的估计值
        严格按照DNV AS文档公式实现
        
        参数:
            data (np.ndarray): 时间序列数据
            t (np.ndarray): 时间戳数组
        
        返回:
            float: 斜率b₁的估计值
        """
        n = len(data)
        if n < 2:
            raise ValueError("数据点数量必须至少为2")
        
        # 严格按照DNV AS文档公式实现
        # b₁ = Σ(t_i)(x_i) - (Σt_i)(Σx_i)/n / Σ(t_i²) - (Σt_i)²/n
        sum_t = np.sum(t)
        sum_x = np.sum(data)
        sum_tx = np.sum(t * data)
        sum_t_squared = np.sum(t ** 2)
        
        numerator = sum_tx - (sum_t * sum_x) / n
        denominator = sum_t_squared - (sum_t ** 2) / n
        
        # 避免除以零
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def _calculate_intercept(self, data: np.ndarray, t: np.ndarray, b1: float) -> float:
        """
        计算截距b₀的估计值
        严格按照DNV AS文档公式实现
        
        参数:
            data (np.ndarray): 时间序列数据
            t (np.ndarray): 时间戳数组
            b1 (float): 斜率估计值
        
        返回:
            float: 截距b₀的估计值
        """
        # 严格按照DNV AS文档公式实现
        # b₀ = (1/n)Σx_i - b₁*(1/n)Σt_i
        n = len(data)
        sum_x = np.sum(data)
        sum_t = np.sum(t)
        
        return (sum_x / n) - b1 * (sum_t / n)
    
    def _estimate_noise_std(self, data: np.ndarray, t: np.ndarray, 
                           b1: float, b0: float) -> float:
        """
        估计白噪声标准差σₐ̂
        严格按照DNV AS文档公式实现
        
        参数:
            data (np.ndarray): 时间序列数据
            t (np.ndarray): 时间戳数组
            b1 (float): 斜率估计值
            b0 (float): 截距估计值
        
        返回:
            float: 白噪声标准差的估计值
        """
        n = len(data)
        if n <= 2:
            raise ValueError("数据点数量必须大于2以计算噪声标准差")
        
        # 严格按照DNV AS文档公式实现
        # σₐ̂ = sqrt(Σ(x_i - (b₀ + b₁t_i))² / (n-2))
        residuals = data - (b0 + b1 * t)
        sum_squared_residuals = np.sum(residuals ** 2)
        
        sigma_a_hat = np.sqrt(sum_squared_residuals / (n - 2))
        
        return sigma_a_hat
    
    def _estimate_slope_std(self, data: np.ndarray, t: np.ndarray, 
                           b1: float, b0: float) -> float:
        """
        估计斜率的标准差σᵦ₁̂
        严格按照DNV AS文档公式实现
        
        参数:
            data (np.ndarray): 时间序列数据
            t (np.ndarray): 时间戳数组
            b1 (float): 斜率估计值
            b0 (float): 截距估计值
        
        返回:
            float: 斜率标准差的估计值
        """
        n = len(t)
        
        # 严格按照DNV AS文档公式实现
        # σᵦ₁̂ = σₐ̂ / sqrt(Σ(t_i - (1/n)Σt_i)²)
        # 直接计算Σ(t_i - (1/n)Σt_i)²，不使用转换公式
        mean_t = np.mean(t)
        denominator = np.sum((t - mean_t) ** 2)
        
        if denominator == 0:
            return 0
        
        # 获取白噪声标准差
        sigma_a_hat = self._estimate_noise_std(data, t, b1, b0)
        
        sigma_b1_hat = sigma_a_hat / np.sqrt(denominator)
        
        return sigma_b1_hat
    
    def perform_test(self, data: Union[List, np.ndarray], 
                    time_stamps: Optional[Union[List, np.ndarray]] = None) -> Dict:
        """
        执行线性斜率检验
        
        参数:
            data (Union[List, np.ndarray]): 要测试的时间序列数据
            time_stamps (Optional[Union[List, np.ndarray]]): 对应的时间戳，如果为None则使用0到n-1
        
        返回:
            Dict: 包含测试结果的字典，包括:
                - is_steady: 是否为稳态
                - b1_hat: 斜率估计值
                - b0_hat: 截距估计值
                - t_statistic: t统计量
                - t_critical: 临界t值
                - p_value: p值
                - degrees_of_freedom: 自由度
                - sigma_a_hat: 白噪声标准差估计值
                - sigma_b1_hat: 斜率标准差估计值
        """
        # 转换输入为numpy数组
        data = np.array(data)
        if len(data) < 3:
            raise ValueError("数据点数量必须至少为3以执行有效检验")
        
        # 如果没有提供时间戳，使用0到n-1
        if time_stamps is None:
            t = np.arange(len(data))
        else:
            t = np.array(time_stamps)
            if len(t) != len(data):
                raise ValueError("时间戳长度必须与数据长度一致")
        
        n = len(data)
        
        # 计算斜率和截距
        b1_hat = self._calculate_slope(data, t)
        b0_hat = self._calculate_intercept(data, t, b1_hat)
        
        # 估计斜率标准差
        sigma_b1_hat = self._estimate_slope_std(data, t, b1_hat, b0_hat)
        
        # 计算t统计量
        if sigma_b1_hat > 0:
            t_statistic = b1_hat / sigma_b1_hat
        else:
            t_statistic = 0
        
        # 计算自由度和临界值
        df = n - 2  # 自由度
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        
        # 计算p值（双侧检验）
        p_value = 2 * stats.t.sf(abs(t_statistic), df)
        
        # 判断是否稳态
        # 当|t统计量| <= 临界值时，接受原假设（斜率为0），认为数据是稳态的
        is_steady = abs(t_statistic) <= t_critical
        
        return {
            'is_steady': is_steady,
            'b1_hat': b1_hat,
            'b0_hat': b0_hat,
            't_statistic': t_statistic,
            't_critical': t_critical,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'sigma_b1_hat': sigma_b1_hat,
            'n_samples': n,
            'alpha': self.alpha
        }
    
    def sliding_window_test(self, data: Union[List, np.ndarray], 
                          window_length: int = 30, 
                          time_stamps: Optional[Union[List, np.ndarray]] = None, 
                          steady_threshold: float = 0.8) -> Dict:
        """
        执行滑动窗口线性斜率检验
        
        参数:
            data (Union[List, np.ndarray]): 要测试的时间序列数据
            window_length (int): 滑动窗口长度
            time_stamps (Optional[Union[List, np.ndarray]]): 对应的时间戳
            steady_threshold (float): 判定为稳态的窗口比例阈值
        
        返回:
            Dict: 包含滑动窗口测试结果的字典
        """
        data = np.array(data)
        n = len(data)
        
        if window_length > n:
            raise ValueError("窗口长度不能大于数据长度")
        
        if time_stamps is not None:
            time_stamps = np.array(time_stamps)
            if len(time_stamps) != n:
                raise ValueError("时间戳长度必须与数据长度一致")
        
        # 计算窗口数量
        n_windows = n - window_length + 1
        window_results = []
        
        # 对每个窗口执行检验
        for i in range(n_windows):
            window_data = data[i:i+window_length]
            
            # 获取窗口对应的时间戳
            if time_stamps is not None:
                window_time = time_stamps[i:i+window_length]
            else:
                window_time = None
            
            # 执行检验
            result = self.perform_test(window_data, window_time)
            result['window_start'] = i
            result['window_end'] = i + window_length - 1
            window_results.append(result)
        
        # 计算整体统计信息
        steady_windows = sum(1 for r in window_results if r['is_steady'])
        steady_percentage = (steady_windows / n_windows) * 100
        is_overall_steady = steady_percentage >= steady_threshold * 100
        
        # 计算每个数据点的稳态概率
        point_steady_counts = np.zeros(n)
        point_steady_probabilities = np.zeros(n)
        
        for i, result in enumerate(window_results):
            start, end = result['window_start'], result['window_end']
            point_steady_counts[start:end+1] += 1
            if result['is_steady']:
                point_steady_probabilities[start:end+1] += 1
        
        # 避免除以零
        valid_points = point_steady_counts > 0
        point_steady_probabilities[valid_points] /= point_steady_counts[valid_points]
        
        return {
            'overall_steady_state': is_overall_steady,
            'steady_windows': steady_windows,
            'total_windows': n_windows,
            'steady_percentage': steady_percentage,
            'window_length': window_length,
            'steady_threshold': steady_threshold,
            'window_results': window_results,
            'point_steady_probabilities': point_steady_probabilities.tolist()
        }


def perform_linear_slope_test(data: Union[List, np.ndarray], 
                             time_stamps: Optional[Union[List, np.ndarray]] = None,
                             alpha: float = 0.01) -> Dict:
    """
    便捷函数：执行线性斜率检验
    
    参数:
        data (Union[List, np.ndarray]): 要测试的时间序列数据
        time_stamps (Optional[Union[List, np.ndarray]]): 对应的时间戳
        alpha (float): 显著性水平
    
    返回:
        Dict: 测试结果
    """
    tester = LinearSlopeTest(alpha=alpha)
    return tester.perform_test(data, time_stamps)


# 示例用法
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    
    # 1. 生成稳态数据（均值为10，带小随机噪声）
    n = 50
    steady_data = 10 + 0.1 * np.random.randn(n)
    steady_time = np.arange(n)
    
    # 2. 生成非稳态数据（包含线性趋势）
    trend_data = 10 + 0.05 * np.arange(n) + 0.1 * np.random.randn(n)
    trend_time = np.arange(n)
    
    # 创建测试器实例
    tester = LinearSlopeTest(alpha=0.01)
    
    # 测试稳态数据
    print("\n===== 测试稳态数据 =====\n")
    steady_result = tester.perform_test(steady_data, steady_time)
    print(f"是否稳态: {'是' if steady_result['is_steady'] else '否'}")
    print(f"斜率估计值 b₁: {steady_result['b1_hat']:.6f}")
    print(f"截距估计值 b₀: {steady_result['b0_hat']:.6f}")
    print(f"t统计量: {steady_result['t_statistic']:.4f}")
    print(f"临界t值: {steady_result['t_critical']:.4f}")
    print(f"p值: {steady_result['p_value']:.6f}")
    
    # 测试非稳态数据
    print("\n===== 测试非稳态数据 =====\n")
    trend_result = tester.perform_test(trend_data, trend_time)
    print(f"是否稳态: {'是' if trend_result['is_steady'] else '否'}")
    print(f"斜率估计值 b₁: {trend_result['b1_hat']:.6f}")
    print(f"截距估计值 b₀: {trend_result['b0_hat']:.6f}")
    print(f"t统计量: {trend_result['t_statistic']:.4f}")
    print(f"临界t值: {trend_result['t_critical']:.4f}")
    print(f"p值: {trend_result['p_value']:.6f}")
    
    # 执行滑动窗口测试
    print("\n===== 滑动窗口测试 =====\n")
    mixed_data = np.concatenate([
        10 + 0.1 * np.random.randn(30),       # 稳态段1
        10 + 0.05 * np.arange(20) + 0.1 * np.random.randn(20),  # 趋势段
        12 + 0.1 * np.random.randn(30)        # 稳态段2（不同均值）
    ])
    
    sliding_result = tester.sliding_window_test(mixed_data, window_length=20)
    print(f"整体稳态判定: {'是' if sliding_result['overall_steady_state'] else '否'}")
    print(f"稳态窗口数量: {sliding_result['steady_windows']}/{sliding_result['total_windows']}")
    print(f"稳态窗口百分比: {sliding_result['steady_percentage']:.1f}%")
    print(f"窗口长度: {sliding_result['window_length']}")