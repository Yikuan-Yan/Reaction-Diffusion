"""
flow_field_shear_dynamic.py

此脚本演示如何在一个周期性边界的 200×200 标量场上，应用简单剪切流场并生成动画。
流动直接通过更新原始数组 Z（基于 np.roll + 线性插值）来实现，无需显式记录回溯索引。
依赖: numpy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 网格与初始场设置 #############################################################
nx, ny = 200, 200
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
X, Y = np.meshgrid(x, y)
# 原始标量场 Z，将在每一帧更新
Z = np.sin(3 * np.pi * X) * np.cos(3 * np.pi * Y)

# 绘图初始化 ###################################################################
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    Z,
    cmap='viridis',
    origin='lower',
    interpolation='bilinear'
)
ax.set_title('动态剪切流场 — 更新原数组')
ax.axis('off')

# 流场参数 #####################################################################
gamma_dot = 0.5    # 剪切速率
dt_step = 0.02     # 每帧时间步长

# 更新函数 ####################################################################
def update(frame):
    global Z
    # 1) 计算每行的水平移动（像素单位）
    #    物理速度 u = gamma_dot * Y; 物理位移 u*dt_step; 映射到像素: * (nx-1)/2
    shift_phys = gamma_dot * Y[:, 0] * dt_step
    shift_pix = shift_phys * (nx - 1) / 2

    # 2) 按行更新 Z：对每一行应用整数滚动 + 小数线性内插
    Z_new = np.empty_like(Z)
    for j in range(ny):
        row = Z[j]
        # 分离整数位和小数位
        s = shift_pix[j]
        i_int = int(np.floor(s))  # 向下取整
        frac = s - i_int          # 小数部分
        # 整数滚动: -i_int 表示向右移动 i_int 列
        row_int = np.roll(row, -i_int)
        # 相邻列值，用于线性插值
        row_next = np.roll(row_int, -1)
        # 线性插值
        Z_new[j] = (1 - frac) * row_int + frac * row_next

    # 3) 置回原数组
    Z = Z_new
    # 4) 更新图像数据
    im.set_data(Z)
    return im,

# 创建并显示动画 ###############################################################
ani = animation.FuncAnimation(
    fig,
    update,
    frames=200,
    blit=True,
    interval=50
)

plt.show()
