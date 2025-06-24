import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------------------------
# Parameters for the Gray–Scott reaction–diffusion system
# ----------------------------------------------------------------------------
Du = 0.16            # Diffusion rate of chemical U
Dv = 0.08            # Diffusion rate of chemical V
F  = 0.060           # Feed rate of U
k  = 0.062           # Kill rate (removal) of V

# ----------------------------------------------------------------------------
# Grid and spatial setup
# ----------------------------------------------------------------------------
grid_size = 200                    # Number of grid points along each dimension
x = np.linspace(-1, 1, grid_size)  # 1D spatial coordinate from -1 to 1
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)           # 2D meshgrid for spatial calculations

# ----------------------------------------------------------------------------
# Time settings
# ----------------------------------------------------------------------------
dt = 1.0               # Time step for reaction–diffusion updates
total_steps = 20000    # Total number of reaction–diffusion iterations
steps_per_frame = 50    # How many iterations per animation frame

# ----------------------------------------------------------------------------
# Shear flow (advection) parameters
# ----------------------------------------------------------------------------
gamma_dot = 0.5        # Shear rate (velocity gradient)
dt_step = 0.02         # Physical time per frame for the shear displacement

# ----------------------------------------------------------------------------
# Laplacian operator with periodic boundary conditions
# ----------------------------------------------------------------------------
def laplacian(Z):
    """
    Compute the five-point Laplacian of a 2D array Z using periodic boundaries.
    ∇²Z ≈ Z[i-1, j] + Z[i+1, j] + Z[i, j-1] + Z[i, j+1] - 4 * Z[i, j]
    """
    # Create index arrays for vectorized neighbor lookup
    i = np.arange(grid_size)
    j = np.arange(grid_size)
    i_up    = (i - 1) % grid_size  # wrap-around index upward
    i_down  = (i + 1) % grid_size  # wrap-around index downward
    j_left  = (j - 1) % grid_size  # wrap-around index leftward
    j_right = (j + 1) % grid_size  # wrap-around index rightward

    # Sum neighbors minus 4 times center value
    return (
        Z[i_up][:, j] + Z[i_down][:, j]
      + Z[i][:, j_left] + Z[i][:, j_right]
      - 4 * Z
    )

# ----------------------------------------------------------------------------
# Initial conditions: U starts uniform, V has sinusoidal stripes + noise
# ----------------------------------------------------------------------------
def init_conditions(size, stripe_freq=6, noise_amp=0.02):
    """
    Initialize U and V arrays.
    U = 1 everywhere.
    V = 0.25 * (1 + 0.5 * sin(stripe_freq * x)) + small random noise.
    """
    # U starts at concentration 1.0
    U = np.ones((size, size))

    # Create a horizontal sine wave pattern for V
    x_lin = np.linspace(0, 2 * np.pi, size)
    X_tile = np.tile(x_lin, (size, 1))
    V = 0.25 * (1 + 0.5 * np.sin(stripe_freq * X_tile))

    # Add uniform random noise around zero
    V += noise_amp * (np.random.rand(size, size) - 0.5)
    return U, V

# Generate initial U, V
U, V = init_conditions(grid_size)

# ----------------------------------------------------------------------------
# Set up the matplotlib figure and axes for animation
# ----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display for U
caxU = ax1.imshow(U, cmap='inferno', interpolation='bilinear', vmin=0, vmax=1)
ax1.set_title('U concentration')
ax1.axis('off')

# Display for V
caxV = ax2.imshow(V, cmap='inferno', interpolation='bilinear', vmin=0, vmax=1)
ax2.set_title('V concentration')
ax2.axis('off')

# ----------------------------------------------------------------------------
# Shear flow function: advect field Z horizontally with shear + interpolation
# ----------------------------------------------------------------------------
def flow(Z, frame):
    """
    Apply shear flow (advection) to the 2D field Z.
    Each row j is shifted by gamma_dot * Y[j,0] * dt_step in physical units,
    converted to pixel shift, then applied using integer roll + linear interpolation.
    """
    # 1) Compute physical shift for each row: u = gamma_dot * Y
    shift_phys = gamma_dot * Y[:, 0] * dt_step
    # Convert from physical units [-1,1] to pixel units [-(N-1)/2, (N-1)/2]
    shift_pix = shift_phys * (grid_size - 1) / 2

    # Prepare new array
    Z_new = np.empty_like(Z)

    # 2) For each row, apply integer roll + fractional interpolation
    for j in range(grid_size):
        row = Z[j]
        s = shift_pix[j]
        i_int = int(np.floor(s))     # integer part of shift
        frac  = s - i_int            # fractional part for interpolation

        # Roll by integer amount (negative means shift right)
        row_int = np.roll(row, -i_int)
        # Next column for interpolation
        row_next = np.roll(row_int, -1)
        # Linear interpolation between neighbors
        Z_new[j] = (1 - frac) * row_int + frac * row_next

    # 3) Return the advected field
    return Z_new

# ----------------------------------------------------------------------------
# Animation update function
# ----------------------------------------------------------------------------
def update(frame):
    global U, V
    # 1) Reaction-diffusion steps
    for _ in range(steps_per_frame):
        Lu  = laplacian(U)          # diffusion term for U
        Lv  = laplacian(V)          # diffusion term for V
        UV2 = U * V * V             # reaction term U + 2V -> 3V

        # Update U, V according to Gray–Scott equations
        U += (Du * Lu - UV2 + F * (1 - U)) * dt
        V += (Dv * Lv + UV2 - (F + k) * V) * dt

        # Clamp concentrations to [0,1] for stability
        np.clip(U, 0, 1, out=U)
        np.clip(V, 0, 1, out=V)

    # 2) Apply shear flow (advection)
    U = flow(U, frame)
    V = flow(V, frame)

    # 3) Update the images in the plot
    caxU.set_data(U)
    caxV.set_data(V)
    fig.suptitle(f"Step {frame * steps_per_frame}")

    return caxU, caxV

# ----------------------------------------------------------------------------
# Run the animation
# ----------------------------------------------------------------------------
frames = total_steps // steps_per_frame
anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
plt.show()
