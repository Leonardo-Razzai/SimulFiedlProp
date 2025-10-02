from simulation import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Precompute fields & fidelity
z_vals = np.array(z_vals).flatten()
E_list = [propagate_field(z) for z in z_vals]
dx = dy = (2*grid_extent)/grid_size
F_vals = np.array([fidelity(Ez, E_in, dx, dy) for Ez in E_list]).flatten()

# Setup figure
fig, axes = plt.subplots(1, 2, figsize=(10,4))

# Left: intensity
intensity0 = np.abs(E_list[0])**2
intensity0 /= intensity0.max()
im_intensity = axes[0].imshow(intensity0,
                              extent=[-grid_extent, grid_extent, -grid_extent, grid_extent],
                              cmap='inferno', origin='lower')
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Intensity")

# Right: fidelity
line_fid, = axes[1].plot([], [], 'b-', lw=2)
point_fid, = axes[1].plot([], [], 'ro', markersize=6)
axes[1].set_xlim(z_vals[0], z_vals[-1])
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel("z")
axes[1].set_ylabel("Fidelity")
axes[1].set_title("Fidelity with Input")

plt.tight_layout()

# Update function
def update(frame):
    # Update intensity
    E = E_list[frame]
    intensity = np.abs(E)**2
    intensity /= intensity.max()
    im_intensity.set_array(intensity)
    axes[0].set_title(f"Intensity (z={z_vals[frame]:.2f})")

    # Update fidelity
    xdata = np.atleast_1d(z_vals[:frame+1])
    ydata = np.atleast_1d(F_vals[:frame+1])
    line_fid.set_data(xdata, ydata)

    point_fid.set_data([z_vals[frame]], [F_vals[frame]])

    return [im_intensity, line_fid, point_fid]

# Animation
anim = FuncAnimation(fig, update, blit=True, save_count=len(E_list))

# Save GIF
writer = PillowWriter(fps=5)
anim.save("intensity_fidelity_plot.gif", writer=writer)
plt.close(fig)
print("GIF saved as intensity_fidelity_plot.gif")
