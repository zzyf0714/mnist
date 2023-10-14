import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(0, 10, 100)  # 在0到10之间生成100个等间距的点
y = np.sin(x)  # 计算对应x值的正弦值

# 绘制折线图
plt.plot(x, y, label='sin(x)')  # x作为横坐标，y作为纵坐标绘制折线
plt.xlabel('x')  # 设置横坐标标签
plt.ylabel('y')  # 设置纵坐标标签
plt.title('Sine Function')  # 设置图标题
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格线
plt.show()  # 显示图形