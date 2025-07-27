import numpy as np
import matplotlib.pyplot as plt

# Lattice size
Lx, Ly = 100, 100
K = np.zeros((Lx, Ly))  # initial field ∇𝒦
K[Lx//2, Ly//2] = 1  # initial coherence peak at center

# Simulation parameters
alpha = 0.01  # diffusion coefficient
steps = 200
frames = []

# Evolution loop
for t in range(steps):
    laplacian = (
        np.roll(K, 1, axis=0) + np.roll(K, -1, axis=0) +
        np.roll(K, 1, axis=1) + np.roll(K, -1, axis=1) - 4 * K
    )
    K += alpha * laplacian
    if t % 20 == 0:
        frames.append(K.copy())

# Final plot
plt.imshow(K, cmap="inferno")
plt.title("Final ∇𝒦 field distribution")
plt.colorbar(label="coherence")
plt.show()
