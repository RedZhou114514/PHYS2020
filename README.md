# Molecular Dynamics Simulation of Lennard-Jones Fluid

## Overview

This project performs molecular dynamics (MD) simulations based on the Lennard-Jones (LJ) potential to model intermolecular interactions [1]. Using computational physics techniques such as the Velocity Verlet algorithm and periodic boundary conditions, these simulations model a particle system governed by the LJ potential. The project aims to demonstrate particle interaction behaviors and explore fundamental phenomena in thermodynamic statistical physics, specifically velocity distributions and Brownian motion, thereby enhancing the understanding of the connection between microscopic systems and macroscopic properties. All simulations are performed in Python 3.12, with performance enhancements using Numba.

## Physics Background

### Lennard-Jones Potential
The LJ potential describes pairwise interactions, featuring repulsion at very close distances, attraction at medium distances, and no interaction at infinite distances. It is given by:
$V(r) = 4\epsilon[(\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^{6}]$
where $r$ is the interparticle distance, $\epsilon$ is the potential well depth, and $\sigma$ is the distance where the potential is zero. The force is $\vec{F}(r) = -\nabla V(r)$.

### Velocity Distribution
The Maxwell-Boltzmann distribution describes particle speeds in idealized gases at thermodynamic equilibrium:
$f(v) = 4\pi v^{2}(\frac{m}{2\pi k_{B}T})^{\frac{3}{2}}\exp(-\frac{mv^{2}}{2k_{B}T})$
where $v$ is speed, $m$ is mass, $k_B$ is Boltzmann's constant, and $T$ is temperature.

### Brownian Motion
Brownian motion is the random motion of larger particles suspended in a fluid due to collisions with smaller fluid particles. The mean squared displacement (MSD) is given by:
$\langle r^{2}(t)\rangle = 6Dt$
where $D$ is the diffusion coefficient and $t$ is time. $D$ is related to temperature by $D = \mu k_{B}T$ (Einstein relation).

## Simulation Details

* **Units:** Standard Lennard-Jones units are used ($\epsilon=1, \sigma=1$, particle mass $m=1$ for solvent particles, $k_B=1$).
* **Integration:** Velocity Verlet algorithm with $\Delta t = 0.001 \tau_{LJ}$.
* **Boundary Conditions:** Periodic Boundary Conditions (PBC) in a cubic box.
* **Potential Cutoff:** $r_{cut} = 2.5\sigma$.
* **Initialization:** Particles are typically initialized on a Simple Cubic (SC) lattice. Initial velocities are drawn from a Gaussian distribution and scaled to the target temperature.
* **Thermostat:** A Berendsen thermostat is used to maintain system temperature, with a coupling constant $\tau$ (e.g., $0.1 \tau_{LJ}$ or $0.5 \tau_{LJ}$ depending on the script).
* **Language & Acceleration:** Python 3.12 with Numba for JIT compilation of computationally intensive functions.

## Scripts

This project consists of three main Python scripts:

1.  **`Velocity Distribution.py`**
    * **Purpose:** Simulates a system of $N=100$ identical particles to study their velocity distribution at a target temperature (e.g., $T=1.0$).
    * **Key Outputs:**
        * A plot of the simulated velocity distribution histogram compared with the theoretical Maxwell-Boltzmann distribution.
        * A plot of temperature vs. time to monitor equilibration.

2.  **`Brownian motion multiple times.py`**
    * **Purpose:** Simulates Brownian motion by introducing one heavy particle ($m_0=10$) into a system of $N-1=99$ lighter solvent particles at a single target temperature (e.g., $T=1.0$). It performs multiple independent runs (e.g., $N_{runs}=20$ or $50$) to obtain a statistically robust estimate of the diffusion coefficient ($D$).
    * **Key Outputs:**
        * For each run (optional): MSD vs. time plot with linear fit yielding $D$.
        * A scatter plot showing the distribution of calculated $D$ values from all runs, along with the mean $D$, standard deviation ($\sigma_D$), and standard error of the mean (SEM).
        * A plot of temperature vs. time for the last run.

3.  **`Diffusion coefficient vs temperature.py`**
    * **Purpose:** Investigates the relationship between the diffusion coefficient ($D$) and temperature ($T$). It runs multiple simulations (e.g., $N_{runs\_per\_T}=2$, $10$, or $50$) for various target temperatures (e.g., $T=1.0$ to $10.0$).
    * **Key Outputs:**
        * A plot of the mean diffusion coefficient $\bar{D}(T)$ versus temperature $T$, with error bars representing the SEM.
        * A theoretical power-law fit ($D = AT^\gamma$) to the $\bar{D}(T)$ data.

## Dependencies

* Python 3.x
* NumPy
* Matplotlib
* Numba
* SciPy (for `linregress` and `curve_fit`)

You can install these using pip:
`pip install numpy matplotlib numba scipy`

## How to Run

1.  Ensure all dependencies are installed.
2.  Open a terminal or command prompt.
3.  Navigate to the directory containing the scripts.
4.  Execute the desired script using Python:
    * `python "Velocity Distribution.py"`
    * `python "Brownian motion multiple times.py"`
    * `python "Diffusion coefficient vs temperature.py"`

Simulation parameters (like $N$, $L$, $T$, `steps`, `N_runs`, `tau`, temperature range) can be modified directly within each script. Be aware that simulations involving many runs or high temperatures can be computationally intensive.

## Expected Outputs

Each script will generate one or more plots:
* **Velocity Distribution:** A histogram of particle speeds compared to the Maxwell-Boltzmann curve, and a temperature vs. time plot.
* **Brownian Motion (Multiple Runs, Single T):** A scatter plot of individual $D$ values, indicating the mean and spread, and a temperature vs. time plot for the last run. Individual MSD plots per run can be enabled in the code.
* **Diffusion Coefficient vs. Temperature:** A plot of $\bar{D}$ with error bars against $T$, along with a power-law fit.

The scripts will also print progress and final results (e.g., calculated $D$ values, fit parameters) to the console.

## AI Usage Declaration 🤖

Portions of the Python code in this project, particularly related to algorithm implementation (such as lattice initialization, Berendsen thermostat application, Mean Squared Displacement calculation including trajectory unwrapping, and multi-run averaging logic), code optimization strategies with Numba, plotting enhancements (e.g., adaptive ticks, legend formatting), and debugging, were developed with the assistance of an AI programming assistant (Google Gemini). The core simulation concepts and physical models were based on established computational physics principles as outlined in the project's background.

All AI-generated or AI-assisted code segments were carefully reviewed, understood, tested, and adapted by the author to ensure correctness and suitability for the project's objectives. The author remains fully responsible for the overall functionality and integrity of the simulations and the scientific interpretation of their results.

---

Feel free to adjust any section to better reflect the specifics of your project or the emphasis you want to place. For example, you might want to list the exact parameter values you used for the figures in your report directly in the README or refer to the report for those details.
