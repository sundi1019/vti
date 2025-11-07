import math
import numpy as np

class WaveDirectionalSpreadFunction:
    """
    波浪方向分布函数D(α)计算类
    
    该类实现了公式(4.5.2节)中的波浪方向分布函数，
    用于计算波浪能量在不同方向上的分布。
    """
    
    def __init__(self, alpha_p=None, n=None):
        """
        初始化波浪方向分布函数类
        
        参数:
        alpha_p: 主相对波浪方向(Primary relative wave direction)，单位为弧度(rad)
        n: 扩展参数(Spreading parameter)，经验值，对于风浪通常取2或4，对于涌浪可取8
        """
        self.alpha_p = alpha_p  # 主相对波浪方向
        self.n = n  # 扩展参数
        self.alpha = None  # 存储最后计算的角度（非数组）
        self.spread_value = None  # 存储最后计算的分布值（非数组）
        self.validate_parameters()  
    
    def validate_parameters(self):
        """
        验证计算所需的参数是否已设置
        
        异常:
        ValueError: 当必要参数未设置时抛出
        """
        if self.alpha_p is None:
            raise ValueError("参数alpha_p(主相对波浪方向)未设置")
        if self.n is None:
            raise ValueError("参数n(扩展参数)未设置")
    
    def calculate(self, alpha):
        """
        计算波浪方向分布函数值D(α)
        
        参数:
        alpha: 相对浪向α，单位为弧度(rad)
        
        返回:
        D_alpha: 对应角度α的波浪方向分布函数值D(α)
        
        异常:
        ValueError: 当必要参数未设置时抛出
        TypeError: 当alpha不是数值类型时抛出
        """        
        # 确保alpha是数值类型
        if not isinstance(alpha, (int, float, np.number)):
            raise TypeError("参数alpha必须是数值类型")
        
        # 计算角度差的绝对值
        alpha_diff = abs(alpha - self.alpha_p)
        
        # 根据公式计算D(α)
        if alpha_diff <= math.pi / 2:
            # 计算伽马函数项: Γ(1+n/2)/(√πΓ(1/2+n/2))
            gamma_term = math.gamma(1 + self.n/2) / (math.sqrt(math.pi) * math.gamma(0.5 + self.n/2))
            # 计算余弦项: cos^n(α - α_p)
            cos_term = math.cos(alpha - self.alpha_p) ** self.n
            # 计算D(α)
            D_alpha = gamma_term * cos_term
        else:
            # 当|α - α_p| > π/2时，D(α) = 0
            D_alpha = 0.0
        
        # 保存最后计算的值（非数组形式）
        self.alpha = alpha
        self.spread_value = D_alpha
        
        return D_alpha
    
    def get_results(self):
        """
        获取计算结果
        
        返回:
        tuple: (alpha, spread_value) - 角度值和对应的分布函数值（非数组）
        
        异常:
        ValueError: 当尚未进行计算时抛出
        """
        if self.spread_value is None:
            raise ValueError("尚未进行分布函数计算，请先调用calculate()方法")
        if self.alpha is None:
            raise ValueError("角度值未保存，请先调用calculate()方法")
            
        return (self.alpha, self.spread_value)


if __name__ == "__main__":
    # 示例1: 使用初始化参数计算特定角度的分布函数值
    # 风浪情况，n=2
    spread1 = WaveDirectionalSpreadFunction(alpha_p=0.0, n=2)
    D1 = spread1.calculate(alpha=0.0)  # 计算主方向上的分布函数值
    
    # 示例2: 使用set_parameters方法设置参数后计算
    spread2 = WaveDirectionalSpreadFunction()
    spread2.set_parameters(alpha_p=math.pi/6, n=4)  # 主方向为30度，风浪情况n=4
    D2 = spread2.calculate(alpha=math.pi/12)  # 计算偏离主方向15度处的分布函数值
    
    # 示例3: 涌浪情况，n=8
    spread3 = WaveDirectionalSpreadFunction(alpha_p=math.pi/4, n=8)
    D3 = spread3.calculate(alpha=math.pi/4)  # 计算主方向上的分布函数值
    
    # 示例4: 计算超出范围的角度
    spread4 = WaveDirectionalSpreadFunction(alpha_p=0.0, n=3)  # IACS Rec.34推荐值n=3
    D4 = spread4.calculate(alpha=math.pi)  # 计算偏离主方向180度处的分布函数值
    
    # 示例5: 获取计算结果
    spread5 = WaveDirectionalSpreadFunction(alpha_p=0.0, n=2)
    D5 = spread5.calculate(alpha=math.pi/6)  # 计算偏离主方向30度处的分布函数值
    alpha, spread_values = spread5.get_results()  # 获取计算结果