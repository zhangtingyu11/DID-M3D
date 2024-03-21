import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import numpy as np

# 定义自定义颜色列表
# colors = [(1, 0, 0, 0), (1, 0, 0, 0.25), (1, 0, 0, 0.5)]  # 红、绿、蓝，以RGB元组形式
# colors = [(1, 0, 0, alpha) for alpha in np.arange(0, 0.5, 0.01)]

# colors = [(0, 0, 1, 0), (0, 0, 1, 0.25), (0, 0, 1, 0.5)]  # 红、绿、蓝，以RGB元组形式
# colors = [(0, 1, 0, 0), (0, 1, 0, 0.25), (0, 1, 0, 0.5)]  # 红、绿、蓝，以RGB元组形式

colors = [(0, 0, 1, alpha) for alpha in np.arange(0, 0.5, 0.01)]
custom_cmap = ListedColormap(colors)

# 创建一个7*7的网格
grid = [[random.randint(1, 255) for _ in range(7)] for _ in range(7)]
grid_size = 7
center_value = 5
for i in range(grid_size):
    for j in range(grid_size):
        distance_to_center = np.sqrt((i - (grid_size-1)/2)**2 + (j - (grid_size-1)/2)**2)
        grid[i][j] = center_value - distance_to_center + np.random.uniform(-1.0, 1.0)  # 添加噪声

plt.figure(figsize=(5, 5))
plt.imshow(grid, cmap=custom_cmap, origin='lower')

# 添加网格线和边框线
for y in range(8):
    plt.axhline(y - 0.5, color='black', linewidth=1)
for x in range(8):
    plt.axvline(x - 0.5, color='black', linewidth=1)
plt.axhline(-0.5, color='black', linewidth=1)
plt.axhline(6.5, color='black', linewidth=1)
plt.axvline(-0.5, color='black', linewidth=1)
plt.axvline(6.5, color='black', linewidth=1)
plt.axis('off')  # 关闭坐标轴
plt.savefig("blue_grid_att.svg", bbox_inches='tight', pad_inches=0)
plt.close()

colors = [(0, 0, 1, alpha) for alpha in np.arange(0.5, 1, 0.01)]
custom_cmap = ListedColormap(colors)

# 创建一个7*7的网格
grid = [[random.randint(1, 255) for _ in range(7)] for _ in range(7)]
grid_size = 7
center_value = 5
for i in range(grid_size):
    for j in range(grid_size):
        distance_to_center = np.sqrt((i - (grid_size-1)/2)**2 + (j - (grid_size-1)/2)**2)
        grid[i][j] = center_value - distance_to_center + np.random.uniform(-1.0, 1.0)  # 添加噪声

plt.figure(figsize=(5, 5))
plt.imshow(grid, cmap=custom_cmap, origin='lower')

# 添加网格线和边框线
for y in range(8):
    plt.axhline(y - 0.5, color='black', linewidth=1)
for x in range(8):
    plt.axvline(x - 0.5, color='black', linewidth=1)
plt.axhline(-0.5, color='black', linewidth=1)
plt.axhline(6.5, color='black', linewidth=1)
plt.axvline(-0.5, color='black', linewidth=1)
plt.axvline(6.5, color='black', linewidth=1)
plt.axis('off')  # 关闭坐标轴
plt.savefig("blue_grid_att_merged.svg", bbox_inches='tight', pad_inches=0)


