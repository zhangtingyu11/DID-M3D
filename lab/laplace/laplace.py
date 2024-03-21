import numpy as np
import matplotlib.pyplot as plt

# 定义拉普拉斯分布的概率密度函数
def laplace_pdf(x, mu, b):
    return 1.0 / (2 * b) * np.exp(-np.abs(x - mu) / b)

def laplace_cdf(x, mu, b):
    if x < mu:
        return 1/2 * np.exp(-(mu-x)/b)
    else:
        return 1-1/2 * np.exp(-(x-mu)/b)
    
print(laplace_cdf(10.1, 9.5, 0.5)-laplace_cdf(9.9, 9.5, 0.5) + laplace_cdf(10.1, 10.5, 0.5)-laplace_cdf(9.9, 10.5, 0.5))
print(laplace_cdf(10.6, 9.5, 0.5)-laplace_cdf(10.4, 9.5, 0.5) + laplace_cdf(10.6, 10.5, 0.5)-laplace_cdf(10.4, 10.5, 0.5))
print(laplace_cdf(9.6, 9.5, 0.5)-laplace_cdf(9.4, 9.5, 0.5) + laplace_cdf(9.6, 10.5, 0.5)-laplace_cdf(9.4, 10.5, 0.5))



# # 生成一些横坐标数据
# x = np.linspace(0, 20, 1000)

# # 参数设置
# params = [(10, 0.5), (10, 1), (10, 2)]  # (mu, b)

# # 创建子图
# fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)

# # 在每个子图上绘制概率密度函数
# for i, (mu, b) in enumerate(params):
#     pdf_values = laplace_pdf(x, mu, b)
#     axs[i].plot(x, pdf_values)
#     # axs[i].axis('off')
    
#     axs[i].set_title(f'Laplace PDF (mu={mu}, b={b})')
#     axs[i].set_xlabel('x')
#     # axs[0].set_ylabel('Probability Density')
#     axs[i].grid(True)
#     axs[i].set_xlim(0, 20)
#     if i != 0:
#         axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

# plt.tight_layout()
# # plt.show()
# plt.savefig("laplace_distribution.svg", bbox_inches='tight', pad_inches=0)
