# VTI 船舶性能预测项目

这是一个用于船舶性能预测的Python项目，包含船舶功率预测、湿表面积计算等功能。

## 项目结构

- `main.py` - 主程序入口
- `config.yaml` - 配置文件
- `predict_power.py` - 功率预测模块
- `predict_wetted_surface.py` - 湿表面积预测模块
- `*.py` - 其他功能模块
- `*.joblib` - 机器学习模型文件

## 功能特性

- 船舶功率预测
- 湿表面积计算
- 基于机器学习的预测模型
- 配置文件管理

## 安装要求

```bash
pip install pyyaml joblib scikit-learn pandas numpy
```

## 使用方法

```bash
python main.py
```

## 模型文件

- `ship_wetted_surface_models.joblib` - 湿表面积预测模型
- `ship_with_PSV_bspline_models.joblib` - 带PSV的B样条模型
- `ship_without_PSV_bspline_models.joblib` - 不带PSV的B样条模型

## 许可证

MIT License