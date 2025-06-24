import numpy as np
import matplotlib.pyplot as plt

N = 100

# Define dimensions
rows, cols = 4*N, N
# Create a 2D array of zeros using numpy.zeros
D = np.zeros((rows, cols, 7), dtype=float)
x = np.zeros(N**2*4, dtype = float)
y = np.zeros(N**2*4, dtype = float)
color_values = np.zeros(N**2*4, dtype = float)
dot = -1
for i in range(rows):
    for j in range(cols):
        dot+=1
        F = 0.0625 / N * (i+1) # F
        k = 0.0625 / N * (j+1) # k
        D[i,j,1] = F
        D[i,j,2] = k

        if 1-4*(F+k)**2/F < 0:
            break
        u1 = (1+(1-4*(F+k)**2/F)**0.5)/2
        v1 = (F + k) / u1
        u2 = (1-(1-4*(F+k)**2/F)**0.5)/2
        v2 = (F + k) / u2
        val1 = 1
        val2 = 1
        #print(F, k, u1, v1)
        if k < v1**2: val1 = 0
        if k < v2**2: val2 = 0
        val = val1+val2
        if val != 0:
            x[dot] = F
            y[dot] = k
            color_values[dot] = val1+val2

# Create a scatter plot
plt.scatter(x, y, c=color_values, cmap='viridis', s=1)
plt.colorbar(label='Color Value')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Scatter Plot with Color')
plt.show()