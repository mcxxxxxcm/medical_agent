import numpy as np
import sys

print(f"Python 执行路径: {sys.executable}")
print(f"NumPy 版本: {np.__version__}")
print(f"NumPy 文件路径: {np.__file__}")

# 尝试访问那个报错的属性
try:
    print(f"np.float_ 存在: {hasattr(np, 'float_')}")
    val = np.float_
    print(f"np.float_ 值: {val}")
except AttributeError as e:
    print(f"❌ 错误: {e}")