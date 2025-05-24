import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.stats import linregress
from scipy.optimize import curve_fit

# --- Parameters ---
N = 100
L = 10.0
V = L ** 3
n = N / V

temperatures_to_simulate = np.arange(1.0, 10.1, 1.0)
N_runs_per_T = 100

dt = 0.001
steps = 10000
r_cut = 2.5
tau = 0.1
sample_interval = 100

# --- Lattice Initialization ---
print("Initializing positions from Simple Cubic (SC) lattice...")
N_per_side = int(np.ceil(N**(1/3.0)))
spacing = L / N_per_side
positions_initial = np.zeros((N, 3))
idx = 0
for x_val in range(N_per_side):
    for y_val in range(N_per_side):
        for z_val in range(N_per_side):
            if idx < N:
                positions_initial[idx, 0] = x_val * spacing
                positions_initial[idx, 1] = y_val * spacing
                positions_initial[idx, 2] = z_val * spacing
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
                F = F_mag * r_vec # LJ Force
                forces[i] += F
                forces[j] -= F
                potential_energy += 4.0 * (inv_r12 - inv_r6) # LJ Potential
    return forces, potential_energy

# --- Store D vs T results ---
mean_D_values = []
sem_D_values = []

# --- Temperature Loop ---
for T_target in temperatures_to_simulate:
    print(f"\n\n========= Simulating for Target Temperature T = {T_target:.1f} =========")
    d_values_current_T = []

    # --- Runs per Temperature Loop ---
    for run in range(N_runs_per_T):
        print(f"\n======= Starting Run {run + 1}/{N_runs_per_T} for T = {T_target:.1f} =======")

        # --- Initialize for each run ---
        positions = positions_initial.copy()
        velocities = np.random.randn(N, 3)
        velocities -= np.mean(velocities, axis=0)
        current_KE = 0.5 * np.sum(masses_col * velocities**2)
        current_T_val = (2.0 * current_KE) / (3.0 * N)
        velocities *= np.sqrt(T_target / current_T_val) # Scale velocities to T_target
        initial_KE = 0.5 * np.sum(masses_col * velocities**2)
        initial_T = (2.0 * initial_KE) / (3.0 * N)
        print(f"Target T: {T_target:.1f}, Initial T set to: {initial_T:.6f}")

        big_particle_positions = []

        # --- MD Step Loop ---
        for step in range(steps):
            forces, potential_energy = calculate_forces(positions, N, L, r_cut)

            accelerations = forces / masses_col
            # --- Verlet Integration ---
            positions += velocities * dt + 0.5 * accelerations * dt**2
            positions %= L # Apply PBC
            new_forces, _ = calculate_forces(positions, N, L, r_cut)
            new_accelerations = new_forces / masses_col
            velocities += 0.5 * (accelerations + new_accelerations) * dt

            kinetic_energy = 0.5 * np.sum(masses_col * velocities**2)
            current_T_val = (2.0 * kinetic_energy) / (3.0 * N)

            # --- Berendsen Thermostat ---
            factor_inside_sqrt = 1.0 + (dt / tau) * (T_target / current_T_val - 1.0)
            scale_factor = np.sqrt(factor_inside_sqrt)
            velocities *= scale_factor

            # --- Data Sampling ---
            if step % sample_interval == 0:
                big_particle_positions.append(positions[0].copy())

            if step % (steps // 4) == 0 and step > 0:
                print(f"  Run {run + 1}, T={T_target:.1f}, Step {step}/{steps}, Inst_T={current_T_val:.3f}")

        print(f"Run {run + 1} for T={T_target:.1f} finished.")

        # --- MSD Calculation (per run) ---
        big_particle_positions_arr = np.array(big_particle_positions)
        unwrapped_positions = np.zeros_like(big_particle_positions_arr)
        unwrapped_positions[0] = big_particle_positions_arr[0]
        for i in range(1, len(big_particle_positions_arr)): # Unwrap trajectory
            delta = big_particle_positions_arr[i] - big_particle_positions_arr[i-1]
            delta -= L * np.round(delta / L)
            unwrapped_positions[i] = unwrapped_positions[i-1] + delta

        displacements_sq = np.sum((unwrapped_positions - unwrapped_positions[0])**2, axis=1) # MSD
        times = np.arange(len(displacements_sq)) * dt * sample_interval

        slope, intercept, r_value, p_value, std_err_slope = linregress(times, displacements_sq) # Fit
        D_current_run = slope / 6.0 # Diffusion Coefficient
        print(f"  Run {run + 1} D = {D_current_run:.4f} (R^2 = {r_value**2:.4f})")
        d_values_current_T.append(D_current_run)

    # --- Aggregate D values for current T ---
    mean_D_for_T = np.mean(d_values_current_T)
    std_D_for_T = np.std(d_values_current_T)
    sem_D_for_T = std_D_for_T / np.sqrt(len(d_values_current_T))
    mean_D_values.append(mean_D_for_T)
    sem_D_values.append(sem_D_for_T)
    print(f"For T = {T_target:.1f}: Mean D = {mean_D_for_T:.4f} +/- {sem_D_for_T:.4f} (SEM from {len(d_values_current_T)} runs)")

# --- Final D vs T Analysis and Plot ---
print("\n\n======= D vs T Results =======")
plot_temps = np.array(temperatures_to_simulate)
plot_mean_D = np.array(mean_D_values)
plot_sem_D = np.array(sem_D_values)

plt.figure(figsize=(10, 6))
plt.errorbar(plot_temps, plot_mean_D, yerr=plot_sem_D, fmt='o', capsize=5, label='Simulated D (Mean +/- SEM)') # Plot D vs T

# --- Theoretical Fit: D = A * T^gamma ---
def power_law(T_fit_arg, A, gamma):
    return A * (T_fit_arg**gamma)

popt, pcov = curve_fit(power_law, plot_temps, plot_mean_D, sigma=plot_sem_D, p0=[0.1, 1.0], bounds=([0, 0.1],[10, 3])) # Fit data
A_fit, gamma_fit = popt
print(f"\nFit results for D = A * T^gamma:")
print(f"  A = {A_fit:.4f}, gamma = {gamma_fit:.4f}")
T_fit_range = np.linspace(min(plot_temps), max(plot_temps), 100)
D_fitted_curve = power_law(T_fit_range, A_fit, gamma_fit)
plt.plot(T_fit_range, D_fitted_curve, 'r-', label=f'Fit: $D = {A_fit:.2f} \cdot T^{{{gamma_fit:.2f}}}$') # Plot fit

plt.xlabel('Temperature T (LJ units)')
plt.ylabel('Diffusion Coefficient D (LJ units)')
plt.title('Diffusion Coefficient vs Temperature')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()