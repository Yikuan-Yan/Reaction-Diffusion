import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Du = 0.16
Dv = 0.08
F = 0.060
k = 0.062

grid_size = 200
dt = 1.0
total_steps = 20000
steps_per_frame = 50

U = np.ones((grid_size, grid_size))
V = np.zeros((grid_size, grid_size))
r = 20
center = grid_size // 2
U[center-r:center+r, center-r:center+r] = 0.50
V[center-r:center+r, center-r:center+r] = 0.25

i = np.arange(grid_size)
j = np.arange(grid_size)
i_up = (i - 1) % grid_size
i_down = (i + 1) % grid_size
j_left = (j - 1) % grid_size
j_right = (j + 1) % grid_size

def laplacian(Z):
    return Z[i_up][:, j] + Z[i_down][:, j] + Z[i][:, j_left] + Z[i][:, j_right] - 4 * Z

fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(U, cmap='inferno', interpolation='bilinear')
ax.axis('off')

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
    cax.set_data(U)
    ax.set_title(f"Step {frame * steps_per_frame}")
    return cax,

frames = total_steps // steps_per_frame
anim = FuncAnimation(fig=fig, func=update, frames=frames, interval=50, blit=False)
plt.show()
