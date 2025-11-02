import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Circle
from scipy.special import jv, kn, jn_zeros
from scipy.optimize import root_scalar


def left_side(l, u):
    return u * (jv(l + 1, u) / jv(l, u))


def right_side(l, V, u):
    w = np.sqrt(V**2 - u**2)
    return w * (kn(l + 1, w) / kn(l, w))


def diff(l, V, u):
    return left_side(l, u) - right_side(l, V, u)


# ------------------------------------------- PARAMETERS ------------------------------------------
# -------------------------------------------------------------------------------------------------

# List of LP modes to compute as (l, m) pairs:
MODES = [(0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]

# Offset added to each mode's cutoff to obtain the working normalized frequency V.
# V = cutoff + V_HIGHER_THAN_CUTOFF. Use a positive value to evaluate the mode.
V_HIGHER_THAN_CUTOFF = 5

# --- Visualization stuff ---
# ---------------------------
# Number of subplot rows when displaying multiple LP mode intensity plots.
GRAPH_N_ROW = 2 

# Axis size (in units of the fiber radius) used for plotting limits.
# The plotting extent will be [-AXIS_SIZE*radius, AXIS_SIZE*radius] in both x and y.
AXIS_SIZE = 1.5

# If True, overlay vector arrows showing the polarization along the central horizontal line.
VECTOR_ARROWS = False

# If True, display diagnostic plots used during root finding:
ROOT_FINDING_VISUALIZATION = False


# --- Saving stuff ---
# --------------------
# Enable/disable saving of generated figures. Set to True to write files to disk.
SAVE = False

# Folder where images will be saved. Relative to the script's working directory.
# It will be created if not existing.
SAVING_FOLDER = Path("Images")

# Suffix appended to the filename when saving (e.g. "LP_mode_intensity_no_arrow.png").
# Use an empty string if no suffix is desired.
SAVE_KWORD = "_no_arrow"

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


SAVING_FOLDER.mkdir(parents=True, exist_ok=True)

# --- Fiber radius ---
radius = 1.0

# --- Grid ---
grid_size = 500
axis_ext = AXIS_SIZE * radius
x = np.linspace(-axis_ext, axis_ext, grid_size)
y = np.linspace(-axis_ext, axis_ext, grid_size)
X, Y = np.meshgrid(x, y)

# --- Ploar coordinates ---
R = np.sqrt(X**2 + Y**2)
PHI = np.arctan2(Y, X)

E_list = []
I_list = []
Names = []

for l, m in MODES:
    # --- Find the cutoff ---

    # Find the cutoff for every mode, with particulary care for l=0 case
    if l == 0 and m == 1:
        cutoff = 1e-6
    elif l == 0:
        cutoff = jn_zeros(1, m - 1)[-1]
    else:
        cutoff = jn_zeros(l - 1, m)[-1]

    V = cutoff + V_HIGHER_THAN_CUTOFF

    # Define the braket of the minimim finding by enclosing the working area in between  the cut off and the asymptote after the cutoff
    bracket_min = cutoff + 1e-6
    bracket_max = min(jn_zeros(l, m)[-1] - 1e-6, V - 1e-6)
    bracket = (bracket_min, bracket_max)

    if ROOT_FINDING_VISUALIZATION:
        fig, ax = plt.subplots(1, 2)
        u = np.linspace(1e-6, V - 1e-6, 1000)

        ax[0].plot(u, left_side(l, u), "o", color="dodgerblue")
        ax[0].plot(u, right_side(l, V, u), "-", color="Black")
        ax[0].hlines(0, 0, V, color="black")
        ax[0].set_ylim(-10, 10)
        ax[0].grid(True)

        ax[1].plot(u, diff(l, V, u), "o", color="darkorange")
        ax[1].hlines(0, 0, V, color="black")
        ax[1].plot(bracket, (0, 0), color="red")
        ax[1].grid(True)
        ax[1].set_ylim(-20, 20)

    plt.show()

    result = root_scalar(f=lambda x: diff(l, V, x), bracket=bracket, method="bisect")
    print("\n", f"Mode LP{l}{m}\n", result)

    u_lp = result.root
    w_lp = np.sqrt(V**2 - u_lp**2)

    B = jv(l, u_lp) / kn(l, w_lp)

    Ey_lp_core = jv(l, u_lp / radius * R) * np.cos(l * PHI)
    Ey_lp_cladding = B * kn(l, w_lp / radius * R) * np.cos(l * PHI)

    # --- I look only the core ---
    Ey_lp_core[R > radius] = 0
    Ey_lp_cladding[R < radius] = 0

    Ey_lp_tot = Ey_lp_core + Ey_lp_cladding

    # --- Intensity ---
    I_lp_tot = Ey_lp_tot**2

    E_list.append(Ey_lp_tot)
    I_list.append(I_lp_tot)

    Names.append(f"LP{l}{m}")


# --- Graph ---
cmap = "gnuplot2"
# gnuplot2 and gnuplot very good cmap, particularly the version with the 2
# jet, inferno not bad

n_col_graph = len(MODES) // GRAPH_N_ROW + (1 if len(MODES) % GRAPH_N_ROW != 0 else 0)

fig, axes = plt.subplots(GRAPH_N_ROW, n_col_graph, sharex=True, sharey=True)
fig.suptitle(r"Comparison of $LP_{lm}$ modes intensity")

for I_mode, E_mode, ax, title in zip(I_list, E_list, axes.flat, Names):
    im = ax.imshow(
        I_mode,
        cmap=cmap,
        vmin=0,
        vmax=I_mode.max(),
        origin="lower",
        aspect="equal",
        extent=[-axis_ext, axis_ext, -axis_ext, axis_ext],
    )

    ax.tick_params(
        axis="both", which="major", labelsize=6
    )  # Adjust the font size of tick labels

    # -- Add colorbar --
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)  # Adjust the font size of the colorbar ticks
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    if VECTOR_ARROWS:
        skip = 20
        y_coords = np.zeros_like(x[::skip])
        y_zero_index = grid_size // 2
        ax.quiver(
            x[::skip],
            y_coords,
            np.zeros_like(E_mode[y_zero_index, ::skip]),
            E_mode[y_zero_index, ::skip],
            color="white",
            pivot="middle",
            scale=2,
        )

    ax.set_title(f"Mode " + title, pad=-20)

    # -- Add fiber core profile --
    circ = Circle((0, 0), radius, edgecolor="white", facecolor="none", linewidth=1)
    circ.set_zorder(2)
    ax.add_patch(circ)

plt.tight_layout()
if SAVE:
    fig.savefig(SAVING_FOLDER / ("LP_mode_intensity" + SAVE_KWORD), dpi=700)

plt.show()
