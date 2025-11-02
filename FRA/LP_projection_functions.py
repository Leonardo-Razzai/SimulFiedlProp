import numpy as np
from scipy.optimize import root_scalar
from scipy.special import jv, kn, jn_zeros

def left_side(l, u):
    return u * (jv(l + 1, u) / jv(l, u))


def right_side(l, V, u):
    w = np.sqrt(V**2 - u**2)
    return w * (kn(l + 1, w) / kn(l, w))


def diff(l, V, u):
    return left_side(l, u) - right_side(l, V, u)


def get_guided_modes(l, m, V, radius, R, PHI, verbose=False):
    """
    Compute a single guided LP(l,m) mode for a step-index fiber and return its transverse
    E field component, which can be used as either E_x or E_y

    Parameters
    ----------
    l : int
            Azimuthal order of the LP mode (nonnegative integer).
    m : int
            Radial mode index (positive integer, m >= 1).
    V : float
            Normalized frequency parameter of the fibre (k0 * a * sqrt(n_core^2 - n_clad^2)).
    radius : float
            Core radius (same length units as values in R).
    R : array_like
            Radial coordinate grid (same shape as PHI). Elements are radii at which the field
            is evaluated. Values greater than `radius` are interpreted as cladding points.
    PHI : array_like
            Azimuthal coordinate grid (radians), same shape as R, used to compute the
            cos(l*phi) and sin(l*phi) angular dependence.
    verbose : bool, optional
            If True, prints intermediate root-finding results. Default is False.
            
    Returns
    -------
    dict or None
            If the mode is guided (V > cutoff) and the root-finding succeeds, returns a
            dictionary with the following keys:
                - "l": int, same as input l
                - "m": int, same as input m
                - "u": float, solved transverse core eigenvalue u
                - "cos": ndarray, E(r,phi) * cos(l*phi) (same shape as R/PHI)
                - "sin": ndarray, E(r,phi) * sin(l*phi) (same shape as R/PHI)
            If V is below the mode cutoff, returns None.
    """
    
    # Calculate the cutoff of the mode
    if l == 0 and m == 1:
        cutoff = 1e-6
    elif l == 0:
        cutoff = jn_zeros(1, m - 1)[-1]
    else:
        cutoff = jn_zeros(l - 1, m)[-1]

    if V <= cutoff:
        return None

    bracket_min = cutoff + 1e-6
    bracket_max = min(jn_zeros(l, m)[-1] - 1e-6, V - 1e-6)
    bracket = (bracket_min, bracket_max)

    result = root_scalar(f=lambda x: diff(l, V, x), bracket=bracket, method="bisect")

    if verbose:
        print("\n", f"Mode LP{l}{m}\n", result)

    u_lp = result.root
    w_lp = np.sqrt(V**2 - u_lp**2)

    B = jv(l, u_lp) / kn(l, w_lp)

    Ey_lp_core = jv(l, u_lp / radius * R)
    Ey_lp_cladding = B * kn(l, w_lp / radius * R)

    # --- Adjust for their respective domains ---
    Ey_lp_core[R > radius] = 0
    Ey_lp_cladding[R < radius] = 0

    Ey_lp_tot = Ey_lp_core + Ey_lp_cladding

    result = {
        "l": l,
        "m": m,
        "u": u_lp,
        "cos": Ey_lp_tot * np.cos(l * PHI),
        "sin": Ey_lp_tot * np.sin(l * PHI),
    }

    return result


def gaussian_electric_field(
    X,
    Y,
    dA,
    z=0.0,
    w0_x=0.5,
    w0_y=None,
    wavelength=1.0,
    x0=0.0,
    y0=0.0,
    theta=0,
    polarization_angle=0.0,
    amplitude=1.0,
):
    """
    Return complex transverse electric field components (E_x, E_y) of an
    elliptical Gaussian beam in the plane at distance z from the waist at z=0.

    Parameters
    - z: propagation distance (same units as waist and wavelength)
    - w0_x, w0_y: waist radii (w0) along x and y (1/e^2 intensity). If
      w0_y is None it's set equal to w0_x.
    - wavelength: vacuum wavelength
    - x0, y0: beam center offset in the transverse plane
    - polarization_angle: angle (rad) of linear polarization measured from x
    - amplitude: peak amplitude at waist (z=0)

    Uses globals X, Y for the transverse grid.
    Returns complex arrays (E_x, E_y).
    """
    if w0_y is None:
        w0_y = w0_x

    k = 2 * np.pi / wavelength

    # Rayleigh ranges for each axis
    zR_x = np.pi * w0_x**2 / wavelength
    zR_y = np.pi * w0_y**2 / wavelength

    # spot sizes at z
    wx = w0_x * np.sqrt(1.0 + (z / zR_x) ** 2)
    wy = w0_y * np.sqrt(1.0 + (z / zR_y) ** 2)

    # curvature radii (avoid division by zero at z=0)
    if abs(z) < 1e-12:
        Rx = np.inf
        Ry = np.inf
    else:
        Rx = z * (1.0 + (zR_x / z) ** 2)
        Ry = z * (1.0 + (zR_y / z) ** 2)

    # Gouy phases
    gouy = np.arctan(z / zR_x) + np.arctan(z / zR_y)

    # coordinates relative to beam center
    Xc = X - x0
    Yc = Y - y0

    # rotation
    Xr = Xc * np.cos(theta) + Yc * np.sin(theta)
    Yr = -Xc * np.sin(theta) + Yc * np.cos(theta)

    # amplitude normalization so power is comparable at different z
    amp_norm = amplitude * np.sqrt((w0_x * w0_y) / (wx * wy))

    # real-space Gaussian envelope
    envelope = np.exp(-(Xr**2) / (wx**2) - (Yr**2) / (wy**2))

    # curvature phase terms (handle infinite curvature as zero quadratic phase)
    phase_quadratic = 0.0
    if np.isfinite(Rx):
        phase_quadratic += -k * (Xr**2) / (2.0 * Rx)
    if np.isfinite(Ry):
        phase_quadratic += -k * (Yr**2) / (2.0 * Ry)

    # longitudinal phase and total phase
    phase_longitudinal = k * z
    total_phase = phase_longitudinal + phase_quadratic - gouy

    field = amp_norm * envelope * np.exp(1j * total_phase)

    Ex = field * np.cos(polarization_angle)
    Ey = field * np.sin(polarization_angle)

    # power is normalized to 1
    P = np.sum((np.abs(Ex) ** 2 + np.abs(Ey) ** 2)) * dA
    Ex /= np.sqrt(P)
    Ey /= np.sqrt(P)

    return Ex, Ey


def get_LP_modes_projection_coefficients(E_input, mode, dA):
    """
    Compute projection coefficients of a two-component electric field onto an LP mode.

    Parameters
    ----------
    E_input : sequence of numpy.ndarray
        Two-component sampled complex electric field given as (E_x, E_y).
    mode : dict
    dA : float
        Area element associated with each grid sample (integration weight).

    Returns
    -------
    dict
        A dictionary with the following entries:
          - "l", "m", "u": mode identifiers copied from the input mode dict.
          - "P_mode" (float): modal power (norm) computed as sum(|E_mode_cos|^2) * dA.
          - "x_cos", "y_cos", "x_sin", "y_sin": projection coefficients (complex or real)
            corresponding to the overlaps of the input field components with the
            cosine and sine mode fields, normalized by P_mode. Coefficients whose
            absolute value is within a numerical tolerance (1e-10) are returned as 0.

    Notes
    -----
    - Overlap integrals are evaluated as sum(conj(E_mode) * E_input_component) * dA.
      Conjugation is applied to support future use of complex mode bases; for real
     -valued LP modes the conjugation has no effect.
    - P_mode is computed using the cosine-mode field only (assumes the cos/sin pair
      are normalized consistently).
    - A small absolute-tolerance (1e-10) is used to treat numerically negligible
      coefficients as zero to avoid spurious tiny complex residues.
    """


    E_input_x = E_input[0]
    E_input_y = E_input[1]

    E_mode_cos = mode.get("cos")
    E_mode_sin = mode.get("sin")

    # Overlap integrals:
    overlap_x_cos = np.sum(np.conj(E_mode_cos) * E_input_x) * dA
    overlap_y_cos = np.sum(np.conj(E_mode_cos) * E_input_y) * dA
    overlap_x_sin = np.sum(np.conj(E_mode_sin) * E_input_x) * dA
    overlap_y_sin = np.sum(np.conj(E_mode_sin) * E_input_y) * dA

    # NOTE: np.conj is not necessary since the lp modes are described by  means of their real base, althoughit is there
    #       for future upgrade in which we consider complex basis

    P_mode = np.sum(np.abs(E_mode_cos) ** 2) * dA

    A_x_cos = overlap_x_cos / P_mode
    A_y_cos = overlap_y_cos / P_mode
    A_x_sin = overlap_x_sin / P_mode
    A_y_sin = overlap_y_sin / P_mode

    tol = 1e-10

    result = {
        "l": mode.get("l"),
        "m": mode.get("m"),
        "u": mode.get("u"),
        "P_mode": P_mode,
        "x_cos": A_x_cos if not np.isclose(A_x_cos, 0, tol) else 0,
        "y_cos": A_y_cos if not np.isclose(A_y_cos, 0, tol) else 0,
        "x_sin": A_x_sin if not np.isclose(A_x_sin, 0, tol) else 0,
        "y_sin": A_y_sin if not np.isclose(A_y_sin, 0, tol) else 0,
    }

    return result