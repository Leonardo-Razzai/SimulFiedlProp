import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, jn_zeros, jv

# -------------------
# Parameters
# -------------------
wavelength = 0.65       # arbitrary units
k0 = 2*np.pi/wavelength
core_radius = 20.0     # fiber radius
grid_size = 200
grid_extent = 15.0     # transverse window size (smaller → zoom in)

# LG input parameters
p, ell = 0, 1
w0 = 10.0               # beam waist (chosen to fill window better)

# Propagation
z_test = 10e3
z_vals = np.linspace(0, z_test, 200)  # z range
beta_LP01, beta_LP11c, beta_LP11s = 1.0*k0, 0.95*k0, 0.95*(1+1e-4)*k0  # sample propagation constants

# -------------------
# Spatial grid
# -------------------
x = np.linspace(-grid_extent, grid_extent, grid_size)
y = np.linspace(-grid_extent, grid_extent, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)

# -------------------
# Input LG mode
# -------------------
def LG_mode(r, phi, p, ell, w0):
    rho = np.sqrt(2)*r/w0
    L = genlaguerre(p, np.abs(ell))(rho**2)
    field = (rho**np.abs(ell)) * L * np.exp(-rho**2/2) * np.exp(1j*ell*phi)
    return field

E_in = LG_mode(R, Phi, p, ell, w0)
E_in /= np.sqrt(np.sum(np.abs(E_in)**2))  # normalize power

# -------------------
# LP modes (circular waveguide approx)
# -------------------
from scipy.special import jv, kv, jn_zeros

n_core=1.0+1e-3
n_clad=1.0

def LP01(r, phi, core_radius=w0):
    # Effective wave numbers
    k0 = 2*np.pi / wavelength
    delta_n2 = n_core**2 - n_clad**2
    u01 = jn_zeros(0, 1)[0]  # first zero of J0

    # Approximate w using u^2 + w^2 = k0^2 a^2 Δn^2
    w01 = np.sqrt((k0*core_radius)**2 * delta_n2 - u01**2)

    E = np.zeros_like(r)
    # Inside core
    inside = (r <= core_radius)
    E[inside] = jv(0, u01 * r[inside]/core_radius)
    # Outside core: decaying
    outside = (r > core_radius)
    E[outside] = jv(0, u01) * kv(0, w01 * (r[outside]-core_radius)/core_radius)

    return E

def LP11_cos(r, phi, core_radius=w0):
    k0 = 2*np.pi / wavelength
    delta_n2 = n_core**2 - n_clad**2
    u11 = jn_zeros(1, 1)[0]
    w11 = np.sqrt((k0*core_radius)**2 * delta_n2 - u11**2)

    E = np.zeros_like(r)
    inside = (r <= core_radius)
    E[inside] = jv(1, u11*r[inside]/core_radius) * np.cos(phi[inside])
    outside = (r > core_radius)
    E[outside] = jv(1, u11) * kv(1, w11*(r[outside]-core_radius)/core_radius) * np.cos(phi[outside])

    return E

def LP11_sin(r, phi, core_radius=w0):
    k0 = 2*np.pi / wavelength
    delta_n2 = n_core**2 - n_clad**2
    u11 = jn_zeros(1, 1)[0]
    w11 = np.sqrt((k0*core_radius)**2 * delta_n2 - u11**2)

    E = np.zeros_like(r)
    inside = (r <= core_radius)
    E[inside] = jv(1, u11*r[inside]/core_radius) * np.sin(phi[inside])
    outside = (r > core_radius)
    E[outside] = jv(1, u11) * kv(1, w11*(r[outside]-core_radius)/core_radius) * np.sin(phi[outside])

    return E

modes = {
    "LP01": LP01(R, Phi),
    "LP11c": LP11_cos(R, Phi),
    "LP11s": LP11_sin(R, Phi)
}

# Normalize modes
for key in modes:
    modes[key] /= np.sqrt(np.sum(np.abs(modes[key])**2))

# -------------------
# Overlap coefficients
# -------------------
coeffs = {}
for key, mode in modes.items():
    coeffs[key] = np.sum(E_in * np.conj(mode))

# -------------------
# Propagation & reconstruction
# -------------------
def propagate_field(z):
    E = (coeffs["LP01"] * modes["LP01"] * np.exp(1j*beta_LP01*z) +
         coeffs["LP11c"]*modes["LP11c"]*np.exp(1j*beta_LP11c*z) +
         coeffs["LP11s"]*modes["LP11s"]*np.exp(1j*beta_LP11s*z))
    return E

# -------------------
# Plot snapshots
# -------------------
def plot_field(E, z, fname=None):
    intensity = np.abs(E)**2
    intensity /= intensity.max()  # normalize to unity
    phase = np.angle(E)

    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    im0 = axes[0].imshow(intensity, extent=[-grid_extent, grid_extent, -grid_extent, grid_extent],
                         cmap='inferno', origin='lower')
    axes[0].set_title(f"**Intensity** (z={z:.2f})", fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(phase, extent=[-grid_extent, grid_extent, -grid_extent, grid_extent],
                         cmap='twilight', origin='lower')
    axes[1].set_title(f"**Phase** (z={z:.2f})", fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=200)
    plt.show()

def fidelity(Ez, E0, dx, dy):
    num = np.abs(np.sum(Ez * np.conj(E0)) * dx * dy)**2
    den = (np.sum(np.abs(Ez)**2)*dx*dy) * (np.sum(np.abs(E0)**2)*dx*dy)
    return num/den

if __name__ == "__main__":
    # Example snapshot
    
    E_test = propagate_field(z_test)
    plot_field(E_test, z_test)

    E_list = [propagate_field(z) for z in z_vals]

    dx = dy = (2*grid_extent) / grid_size
    F_vals = [fidelity(Ez, E_in, dx, dy) for Ez in E_list]

    plt.plot(z_vals, F_vals)
    plt.xlabel("z")
    plt.ylabel("Fidelity with input")
    plt.show()
