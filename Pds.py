import numpy as np


class PdsCalculator:
    """
    传递功率计算器
    
    用于计算船舶传递功率Pds，公式为：Pds = Pm · η_trans
    其中Pm为入参，η_trans为传动效率（默认为0.99）
    """
    
    def __init__(self):
        """
        初始化传递功率计算器
        设置默认传动效率为0.99
        """
        self.pm = None  # 主机功率(kW)
        self.eta_trans = 0.99  # 传动效率，默认值
    
    def set_parameters(self, pm, eta_trans=None):
        """
        设置计算参数
        
        参数:
            pm (float): 主机功率(kW)，必须为正数
            eta_trans (float, optional): 传动效率，默认为0.99，范围通常在0-1之间
        
        异常:
            ValueError: 当参数值不在有效范围内时抛出
        """
        # pm是必传参数，直接进行验证
        if not isinstance(pm, (int, float)) or pm <= 0:
            raise ValueError("主机功率Pm必须是正数")
        self.pm = pm
        
        if eta_trans is not None:
            if not isinstance(eta_trans, (int, float)) or not 0 < eta_trans <= 1:
                raise ValueError("传动效率eta_trans必须在0-1之间")
            self.eta_trans = eta_trans
    
    def calculate(self):
        """
        计算传递功率Pds
        
        返回:
            float: 传递功率Pds(kW)
            
        异常:
            ValueError: 当必要参数未设置时抛出
        """
        # 验证参数是否已设置
        if self.pm is None:
            raise ValueError("主机功率Pm未设置，请先调用set_parameters方法")
        
        # 使用公式计算传递功率: Pds = Pm · η_trans
        pds = self.pm * self.eta_trans
        
        return pds
    
    def calculate_pds(self, pm, eta_trans=None):
        """
        便捷方法：一次性设置参数并计算传递功率
        
        参数:
            pm (float): 主机功率(kW)，必传参数
            eta_trans (float, optional): 传动效率，默认使用实例中的值
        
        返回:
            float: 传递功率Pds(kW)
        """
        self.set_parameters(pm, eta_trans=eta_trans)
        return self.calculate()


# 示例用法
def example():
    """
    示例：展示如何使用PdsCalculator类计算传递功率
    """
    # 创建计算器实例
    calculator = PdsCalculator()
    
    # 设置参数并计算
    pm_value = 5000.0  # 主机功率，单位kW
    calculator.set_parameters(pm=pm_value)
    pds_result = calculator.calculate()
    
    print(f"示例计算结果:")
    print(f"主机功率 Pm = {pm_value} kW")
    print(f"传动效率 η_trans = {calculator.eta_trans}")
    print(f"传递功率 Pds = {pds_result:.2f} kW")
    
    # 使用便捷方法
    pds_result_quick = calculator.calculate_pds(pm=6000.0, eta_trans=0.98)
    print(f"\n便捷方法计算结果:")
    print(f"主机功率 Pm = 6000.0 kW")
    print(f"传动效率 η_trans = 0.98")
    print(f"传递功率 Pds = {pds_result_quick:.2f} kW")


if __name__ == "__main__":
    example()