import numpy as np
import matplotlib.pyplot as plt
# 定义拉普拉斯分布的累积分布函数
def laplace_cdf(x, mu, b):
    return np.where(x < mu, 0.5 * np.exp((x - mu) / b), 1 - 0.5 * np.exp(-(x - mu) / b))
# 生成一些横坐标数据
x = np.linspace(-5, 5, 1000)
# 参数设置
mu = 0
b = 1
# 计算累积分布函数值
cdf_values = laplace_cdf(x, mu, b)
# 绘制累积分布函数图像
plt.plot(x, cdf_values)
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.title('Laplace Cumulative Distribution Function')
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig("laplace_cdf.svg", bbox_inches='tight', pad_inches=0)