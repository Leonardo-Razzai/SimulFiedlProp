import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Circle
from scipy.special import jv, kn, jn_zeros
from scipy.optimize import root_scalar
import pandas as pd

from LP_projection_functions import (
    get_guided_modes,
    gaussian_electric_field,
    get_LP_modes_projection_coefficients,
)

# --------------------------------------- PARAMETERS ----------------------------------------------
# -------------------------------------------------------------------------------------------------
# NOTE: all the length are measured in units of fiber radius

# --- Various Parameters ---
FIBER_V = 6.3
MODES_TO_TEST = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)]

# --- Injected field parameters ---#
LAMBDA = 0.044
E_IN_Z = 0
W0_X = 0.8
W0_Y = 1.1
X0 = 0
Y0 = -0.2
THETA = np.pi / 9
POLARIZATION_ANGLE = 0

# --- Grid stuff ---
AXIS_SIZE = 1.5
GRID_SIZE = 500

# --- Visualization stuff ---
# Colormap name passed to matplotlib for the power density plots
CMAP = "gnuplot2"

# If True, use a common color scale (same vmax) for input field and guided field plots 
# to allow direct visual comparison. If False, each plot scales independently.
NORMALIZE_COLOR_PALETTE = False

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# --- Fiber radius ---
radius = 1.0

# --- Grid ---
axis_ext = AXIS_SIZE * radius
x = np.linspace(-axis_ext, axis_ext, GRID_SIZE)
y = np.linspace(-axis_ext, axis_ext, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# --- Area of a pixel for the integrals ---
dA = (axis_ext * 2 / GRID_SIZE) ** 2

# --- Ploar coordinates ---
R = np.sqrt(X**2 + Y**2)
PHI = np.arctan2(Y, X)


E_input = gaussian_electric_field(
    X,
    Y,
    dA,
    z=E_IN_Z,
    w0_x=W0_X,
    w0_y=W0_Y,
    wavelength=LAMBDA,
    x0=X0,
    y0=Y0,
    theta=THETA,
    polarization_angle=POLARIZATION_ANGLE,
)

guided_modes = []
coefficients = []
for l, m in MODES_TO_TEST:

    mode = get_guided_modes(l, m, FIBER_V, radius, R, PHI)

    if mode is not None:
        guided_modes.append(mode)
        coefficients_res = get_LP_modes_projection_coefficients(E_input, mode, dA)
        coefficients.append(coefficients_res)


# create a pd dataframe, and make (l,m) the index for easier lookup
df_coeff = pd.DataFrame(coefficients)
df_coeff.set_index(["l", "m"], inplace=True)


print("\n" + "*" * 50 + "\n", "SQUARED MODULUS OF COEFFICIENTS", "\n" + "*" * 50)
print(
    (np.abs(df_coeff.iloc[:, 2:]) ** 2).to_string(
        float_format=lambda x: f"{x:.2f}", justify="center", col_space=6
    )
)
print("\n" + "*" * 50 + "\n")


# Construct the guided field
E_guided_x = np.zeros_like(X, dtype=complex)
E_guided_y = np.zeros_like(X, dtype=complex)
sq_sum = 0

for mode in guided_modes:
    if mode is None:
        continue
    l = mode["l"]
    m = mode["m"]

    coeff = df_coeff.loc[l, m]

    E_guided_x_lm = coeff["x_cos"] * mode["cos"] + coeff["x_sin"] * mode["sin"]
    E_guided_y_lm = coeff["y_cos"] * mode["cos"] + coeff["y_sin"] * mode["sin"]

    E_guided_x += E_guided_x_lm
    E_guided_y += E_guided_y_lm

    sq_sum += coeff["P_mode"] * (
        np.abs(coeff["x_cos"]) ** 2
        + np.abs(coeff["x_sin"]) ** 2
        + np.abs(coeff["y_cos"]) ** 2
        + np.abs(coeff["y_sin"]) ** 2
    )


# Get the power of the guided field and the coupling coefficient
E_input_x = E_input[0]
E_input_y = E_input[1]
I_guided = np.abs(E_guided_x) ** 2 + np.abs(E_guided_y) ** 2
I_input = np.abs(E_input_x) ** 2 + np.abs(E_input_y) ** 2

P_input_core = np.sum(I_input[R < radius]) * dA
P_input = np.sum(I_input) * dA
P_guided = np.sum(I_guided) * dA

eta = P_guided / P_input if P_input != 0 else 0.0

print(f"Sum of squared A coeff = {sq_sum:.2f}")
print(f"P_input on the core = {P_input_core:.3f}")
print(f"P_input = {P_input:.2f}")
print(f"P_guided = {P_guided:.2f}")
print(f"Coupling efficiency = {eta:.3f}")


fig, (ax_left, ax) = plt.subplots(1, 2, figsize=(12, 6))

# Use a common color scale for direct comparison
if NORMALIZE_COLOR_PALETTE:
    vmax = max(np.max(I_input), np.max(I_guided))
else:
    vmax = None

im1 = ax_left.imshow(
    I_input,
    extent=[-axis_ext, axis_ext, -axis_ext, axis_ext],
    origin="lower",
    cmap=CMAP,
    aspect="equal",
    vmin=0,
    vmax=vmax,
)
ax_left.set_title("Input power density")
ax_left.set_xlabel("x (radius units)")
ax_left.set_ylabel("y (radius units)")

# draw core circle on left image
core_circle_left = Circle(
    (0, 0),
    radius,
    facecolor="none",
    edgecolor="white",
    linewidth=1.5,
    linestyle="--",
    zorder=5,
)
ax_left.add_patch(core_circle_left)

cbar1 = fig.colorbar(im1, ax=ax_left)
cbar1.set_label("Power density (arb. units)")


im = ax.imshow(
    I_guided,
    extent=[-axis_ext, axis_ext, -axis_ext, axis_ext],
    origin="lower",
    cmap=CMAP,
    aspect="equal",
    vmin=0,
    vmax=vmax,
)
ax.set_title("Total guided power density")
ax.set_xlabel("x (radius units)")
ax.set_ylabel("y (radius units)")

# draw core circle on right image
core_circle_right = Circle(
    (0, 0),
    radius,
    facecolor="none",
    edgecolor="white",
    linewidth=1.5,
    linestyle="--",
    zorder=5,
)
ax.add_patch(core_circle_right)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Power density (arb. units)")
plt.tight_layout()
plt.show()
