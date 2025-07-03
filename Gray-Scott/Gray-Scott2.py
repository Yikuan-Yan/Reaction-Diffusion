import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Du = 0.16
Dv = 0.08
F = 0.060
k = 0.062

grid_size = 200
Fineness = 0.5
CFL_coeff = 0.64
dx = 4*Du / CFL_coeff * Fineness
dt = 4*Du / CFL_coeff * Fineness**2
total_steps = 20000
steps_per_frame = 50

def laplacian(Z):
    i = np.arange(grid_size)
    j = np.arange(grid_size)
    i_up = (i - 1) % grid_size
    i_down = (i + 1) % grid_size
    j_left = (j - 1) % grid_size
    j_right = (j + 1) % grid_size
    return (Z[i_up][:, j] + Z[i_down][:, j] + Z[i][:, j_left] + Z[i][:, j_right] - 4 * Z)/(dx*dx)

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
    caxU.set_data(U)
    caxV.set_data(V)
    fig.suptitle(f"Step {frame * steps_per_frame}")
    return caxU, caxV

frames = total_steps // steps_per_frame
anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
plt.show()