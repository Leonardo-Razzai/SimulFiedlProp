# Optical Fiber LP Mode Projection

This project simulates the coupling of a tilted, elliptical Gaussian beam into a step-index optical fiber. It calculates the fiber's guided Linearly Polarized (LP) modes and uses modal decomposition (via overlap integrals) to determine the coupling efficiency and the power distribution among those modes.

## Core Physics

The simulation is based on the **weakly-guiding approximation** ($n_{\text{core}} \approx n_{\text{clad}}$) for a step-index fiber.

1.  **Guided Modes:** It finds the guided $LP_{lm}$ modes by numerically solving the characteristic equation that arises from applying boundary conditions to the scalar Helmholtz equation. This involves Bessel functions ($J_l$ for the core, $K_l$ for the cladding) and is implemented in `LP_projection_functions.py`.

2.  **Input Beam:** The input field is modeled as a paraxial Gaussian beam. This beam can be elliptical ($w_{0x} \neq w_{0y}$), offset from the fiber axis ($x_0, y_0$), and tilted using 3D Euler angle rotations (roll, pitch, yaw).

3.  **Modal Projection:** The coupling efficiency is determined by projecting the input electric field $\mathbf{E}_{\text{in}}$ onto the basis of the guided mode fields $\mathbf{E}_{lm}$. The complex amplitude coefficient $c_{lm}$ for each mode is calculated using the overlap integral:

    $$
    \\ \\ \\ c_{lm} = \frac{\iint_A \mathbf{E}_{\text{in}}(x,y) \cdot \mathbf{E}_{lm}^*(x,y) \, dA}{\iint_A |\mathbf{E}_{lm}(x,y)|^2 \, dA}
    $$

    The total guided power is the sum of the power in each mode ($P_{lm} \propto |c_{lm}|^2$).

## File Structure

* `LP_projection.py`: The main executable script. This script:
    * Sets all simulation parameters (fiber V-number, beam properties, grid size).
    * Calls functions from the library to generate the input field and guided modes.
    * Performs the modal projection.
    * Prints a report of coefficients and coupling efficiency.
    * Plots the input power density vs. the guided power density.

* `LP_projection_functions.py`: A library of helper functions.
    * `get_guided_modes`: Solves the characteristic equation for a given $LP_{lm}$ mode.
    * `get_tilted_beam_from_incidence`: Generates the 2D complex electric field for the (potentially tilted) Gaussian beam at the fiber plane.
    * `get_LP_modes_projection_coefficients`: Calculates the overlap integrals and projection coefficients for a single mode.

## How to Use

1.  Ensure you have the required dependencies installed.
2.  Configure the simulation parameters in the `PARAMETERS` section of `LP_projection.py`.
3.  Run the main script from your terminal:
    ```bash
    python LP_projection.py
    ```

### Key Parameters (in `LP_projection.py`)

**Note:** All length parameters (waist, offset, wavelength, etc.) are specified in units of the **fiber core radius**.

* `FIBER_V`: The normalized frequency (V-number) of the fiber.
* `MODES_TO_TEST`: A list of `(l, m)` tuples specifying which LP modes to include in the basis.
* `LAMBDA`: Wavelength of the injected beam.
* `DIST_TO_WAIST`: Distance from the beam waist to the fiber input plane.
* `W0_X`, `W0_Y`: Beam waist radii along x and y.
* `X0`, `Y0`: Beam's incidence point (offset) on the fiber plane.
* `ROLL_ANGLE`, `PITCH_ANGLE`, `YAW_ANGLE`: Euler angles (in radians) to define the 's tilt.
* `GRID_SIZE`: The resolution of the simulation grid (e.g., 500 for a 500x500 grid).

## Dependencies

This project requires the following Python libraries:

* `numpy`
* `scipy`
* `matplotlib`
* `pandas`

## Output

Running the script will:

1.  Print a summary of the squared projection coefficients for each mode to the console.
2.  Print a summary of the total input power, total guided power, and the final coupling efficiency ($\eta$).
3.  Generate a `matplotlib` plot showing two subplots:
    * The input beam's power density.
    * The reconstructed guided field's power density (the part of the input beam that is "captured" by the fiber modes).

## Future Improvements (TODO)

* **Normalize Mode Basis:** The current projection coefficients are calculated relative to the power of each mode (`P_mode`). A more standard approach would be to normalize the mode fields $\mathbf{E}_{lm}$ themselves ($\iint |\mathbf{E}_{lm}|^2 dA = 1$). This would make the coefficients $c_{lm}$ more directly interpretable, as $|c_{lm}|^2$ would represent the power coupled into the mode.
