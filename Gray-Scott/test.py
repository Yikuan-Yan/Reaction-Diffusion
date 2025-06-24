import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Du = 0.16
Dv = 0.08
F = 0.060
k = 0.062

grid_size = 200
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)
dt = 1.0
total_steps = 20000
steps_per_frame = 50

# 流场参数 #####################################################################
gamma_dot = 0.5    # 剪切速率
dt_step = 0.02     # 每帧时间步长


def laplacian(Z):
    i = np.arange(grid_size)
    j = np.arange(grid_size)
    i_up = (i - 1) % grid_size
    i_down = (i + 1) % grid_size
    j_left = (j - 1) % grid_size
    j_right = (j + 1) % grid_size
    return Z[i_up][:, j] + Z[i_down][:, j] + Z[i][:, j_left] + Z[i][:, j_right] - 4 * Z

def init_conditions(size, stripe_freq=6, noise_amp=0.02):
    U = np.ones((size, size))
    x = np.linspace(0, 2*np.pi, size)
    X = np.tile(x, (size, 1))
    V = 0.25 * (1 + 0.5 * np.sin(stripe_freq * X))
    V += noise_amp * (np.random.rand(size, size) - 0.5)
    return U, V

U, V = init_conditions(grid_size)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
caxU = ax1.imshow(U, cmap='inferno', interpolation='bilinear', vmin=0, vmax=1)
ax1.set_title('U concentration')
ax1.axis('off')
caxV = ax2.imshow(V, cmap='inferno', interpolation='bilinear', vmin=0, vmax=1)
ax2.set_title('V concentration')
ax2.axis('off')

# 更新函数 ####################################################################
def flow(Z, frame):
    # 1) 计算每行的水平移动（像素单位）
    #    物理速度 u = gamma_dot * Y; 物理位移 u*dt_step; 映射到像素: * (nx-1)/2
    shift_phys = gamma_dot * Y[:, 0] * dt_step
    shift_pix = shift_phys * (grid_size - 1) / 2

    # 2) 按行更新 Z：对每一行应用整数滚动 + 小数线性内插
    Z_new = np.empty_like(Z)
    for j in range(grid_size):
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
    return Z,



def update(frame):
    global U, V
    for _ in range(steps_per_frame):
        Lu = laplacian(U)
        Lv = laplacian(V)
        UV2 = U * V * V
        U += (Du * Lu - UV2 + F * (1 - U)) * dt
        V += (Dv * Lv + UV2 - (F + k) * V) * dt
        np.clip(U, 0, 1, out=U)
        np.clip(V, 0, 1, out=V)
    U = flow(U, frame)
    V = flow(V, frame)
    caxU.set_data(U)
    caxV.set_data(V)
    fig.suptitle(f"Step {frame * steps_per_frame}")
    return caxU, caxV

frames = total_steps // steps_per_frame
anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
plt.show()