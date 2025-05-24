import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.stats import linregress

# --- Parameters ---
N = 100
L = 10.0
V = L ** 3
n = N / V

T = 1.0
dt = 0.001
steps = 10000
r_cut = 2.5
tau = 0.1
sample_interval = 100
N_runs = 50

# --- Lattice Initialization ---
print("Initializing positions from Simple Cubic (SC) lattice...")
N_per_side = int(np.ceil(N**(1/3.0)))
spacing = L / N_per_side
positions_initial = np.zeros((N, 3))
idx = 0
for x in range(N_per_side):
    for y in range(N_per_side):
        for z in range(N_per_side):
            if idx < N:
                positions_initial[idx, 0] = x * spacing
                positions_initial[idx, 1] = y * spacing
                positions_initial[idx, 2] = z * spacing
                idx += 1
            else:
                break
        if idx >= N:
            break
    if idx >= N:
        break
print(f"Lattice initialization done. {idx} particles placed.")

# --- Set Masses ---
masses = np.ones(N)
masses[0] = 10.0
masses_col = masses[:, np.newaxis]

# --- Force Calculation ---
@numba.jit(nopython=True)
def calculate_forces(positions, N, L, r_cut):
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    lower_bound = 1e-6

    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r_vec -= np.round(r_vec / L) * L # PBC
            norm_r = np.linalg.norm(r_vec)

            if norm_r > lower_bound and norm_r < r_cut:
                inv_r = 1.0 / norm_r
                inv_r6 = inv_r**6
                inv_r12 = inv_r6**2
                F_mag = 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r
                F = F_mag * r_vec
                forces[i] += F
                forces[j] -= F
                potential_energy += 4.0 * (inv_r12 - inv_r6)
    return forces, potential_energy

# --- Store D values ---
d_values = []

# --- Multiple Runs Loop ---
for run in range(N_runs):
    print(f"\n======= Starting Run {run + 1}/{N_runs} =======")

    # --- Initialize Velocities & Positions per Run ---
    positions = positions_initial.copy()
    velocities = np.random.randn(N, 3)
    velocities -= np.mean(velocities, axis=0)
    current_KE = 0.5 * np.sum(masses_col * velocities**2)
    current_T = (2.0 * current_KE) / (3.0 * N)
    velocities *= np.sqrt(T / current_T)
    initial_KE = 0.5 * np.sum(masses_col * velocities**2)
    initial_T = (2.0 * initial_KE) / (3.0 * N)
    print(f"Target T: {T}, Initial T set to: {initial_T:.6f}")

    big_particle_positions = []

    # --- Main Simulation Loop (MD Steps) ---
    for step in range(steps):
        forces, potential_energy = calculate_forces(positions, N, L, r_cut)

        accelerations = forces / masses_col
        # --- Verlet Integration ---
        positions += velocities * dt + 0.5 * accelerations * dt**2
        positions %= L
        new_forces, _ = calculate_forces(positions, N, L, r_cut)
        new_accelerations = new_forces / masses_col
        velocities += 0.5 * (accelerations + new_accelerations) * dt

        kinetic_energy = 0.5 * np.sum(masses_col * velocities**2)
        current_T = (2.0 * kinetic_energy) / (3.0 * N)

        # --- Berendsen Thermostat ---
        factor_inside_sqrt = 1.0 + (dt / tau) * (T / current_T - 1.0)
        scale_factor = np.sqrt(factor_inside_sqrt)
        velocities *= scale_factor

        # --- Data Sampling ---
        if step % sample_interval == 0:
            big_particle_positions.append(positions[0].copy())

        if step % 2500 == 0:
            print(f"  Run {run + 1}, Step {step}/{steps}, T={current_T:.3f}")

    print(f"Run {run + 1} finished.")

    # --- MSD Calculation (per run) ---
    big_particle_positions = np.array(big_particle_positions)
    unwrapped_positions = np.zeros_like(big_particle_positions)
    unwrapped_positions[0] = big_particle_positions[0]
    for i in range(1, len(big_particle_positions)):
        delta = big_particle_positions[i] - big_particle_positions[i-1]
        delta -= L * np.round(delta / L)
        unwrapped_positions[i] = unwrapped_positions[i-1] + delta

    displacements_sq = np.sum((unwrapped_positions - unwrapped_positions[0])**2, axis=1)
    times = np.arange(len(displacements_sq)) * dt * sample_interval

    slope, intercept, r_value, p_value, std_err = linregress(times, displacements_sq)
    D = slope / 6.0
    print(f"  Run {run + 1} D = {D:.4f} (R^2 = {r_value**2:.4f})")
    d_values.append(D)


# --- Final Analysis ---
print("\n======= Final Results =======")
d_values = np.array(d_values)
mean_D = np.mean(d_values)
std_D = np.std(d_values)
N_runs_actual = len(d_values)
sem_D = std_D / np.sqrt(N_runs_actual)

print(f"Number of successful runs: {N_runs_actual}")
print(f"Calculated D values: {np.round(d_values, 4)}")
print(f"Mean Diffusion Coefficient (D): {mean_D:.4f}")
print(f"Standard Deviation of D (sigma_D): {std_D:.4f}")
print(f"Standard Error of the Mean (SEM): {sem_D:.4f}")
print(f"Result: D = {mean_D:.4f} +/- {sem_D:.4f} (SEM)")


# --- Plot D values ---
plt.figure(figsize=(10, 6))
run_numbers = np.arange(1, N_runs_actual + 1)
plt.scatter(run_numbers, d_values, marker='o', s=50, label='Calculated D (Each Run)')
plt.axhline(mean_D, color='r', linestyle='--', label=f'Mean D = {mean_D:.4f}')
plt.axhline(mean_D + std_D, color='g', linestyle=':', label=f'Mean +/- $\sigma_D$ ({std_D:.4f})')
plt.axhline(mean_D - std_D, color='g', linestyle=':')
plt.plot([], [], ' ', label=f'SEM = {sem_D:.4f}') # SEM in legend

plt.xlabel('Run Number')
plt.ylabel('Diffusion Coefficient (D)')
plt.title(f'Distribution of Calculated D values ({N_runs_actual} Runs)')

from matplotlib.ticker import MaxNLocator # For adaptive x-ticks
ax = plt.gca()
if N_runs_actual <= 20 and N_runs_actual > 0 :
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=N_runs_actual))
    plt.xticks(run_numbers)
elif N_runs_actual > 0:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))

plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()