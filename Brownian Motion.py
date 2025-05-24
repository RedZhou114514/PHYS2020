import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.stats import linregress

# --- Parameters ---
N = 100
L = 10.0
V = L ** 3
n = N / V

T = 1.0       # Target temperature
dt = 0.001    # Time step
steps = 10000 # Simulation steps
r_cut = 2.5   # Cutoff radius
tau = 0.1     # Thermostat coupling time
sample_interval = 100 # Data sampling interval

# --- Lattice Initialization ---
print("Initializing positions from Simple Cubic (SC) lattice...")
N_per_side = int(np.ceil(N**(1/3.0)))
spacing = L / N_per_side
positions = np.zeros((N, 3))
idx = 0
for x in range(N_per_side):
    for y in range(N_per_side):
        for z in range(N_per_side):
            if idx < N:
                positions[idx, 0] = x * spacing
                positions[idx, 1] = y * spacing
                positions[idx, 2] = z * spacing
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
masses[0] = 10.0  # Set mass of the first particle
masses_col = masses[:, np.newaxis]

# --- Initialize Velocities ---
velocities = np.random.randn(N, 3)
velocities -= np.mean(velocities, axis=0) # Zero center of mass velocity
current_KE = 0.5 * np.sum(masses_col * velocities**2)
current_T = (2.0 * current_KE) / (3.0 * N)
if current_T < 1e-10: current_T = T # Avoid division by zero
velocities *= np.sqrt(T / current_T) # Scale to target T
initial_KE = 0.5 * np.sum(masses_col * velocities**2)
initial_T = (2.0 * initial_KE) / (3.0 * N)
print(f"Target T: {T}, Initial T set to: {initial_T:.6f}")

# --- Force Calculation (Numba JIT) ---
@numba.jit(nopython=True)
def calculate_forces(positions, N, L, r_cut):
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    lower_bound = 1e-6 # Force calculation lower bound

    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r_vec -= np.round(r_vec / L) * L # Periodic Boundary Conditions (PBC)
            norm_r = np.linalg.norm(r_vec)

            if norm_r > lower_bound and norm_r < r_cut: # Check distance range
                inv_r = 1.0 / norm_r
                inv_r6 = inv_r**6
                inv_r12 = inv_r6**2
                F_mag = 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r
                F = F_mag * r_vec # Calculate LJ force
                forces[i] += F
                forces[j] -= F
                potential_energy += 4.0 * (inv_r12 - inv_r6) # Calculate LJ potential
    return forces, potential_energy

# --- Data Storage ---
temperature_history = []
big_particle_positions = []

print("Starting simulation for Brownian Motion...")

# --- Main Simulation Loop ---
for step in range(steps):
    forces, potential_energy = calculate_forces(positions, N, L, r_cut)

    # --- Crash Check ---
    if np.isnan(forces).any() or np.isinf(forces).any():
        print(f"CRASH @ Step {step}: NaN/Inf detected in forces!")
        break
    if np.isnan(velocities).any() or np.isinf(velocities).any():
        print(f"CRASH @ Step {step}: NaN/Inf detected in velocities!")
        break

    # --- Verlet Integrator ---
    accelerations = forces / masses_col # a = F/m
    positions += velocities * dt + 0.5 * accelerations * dt**2 # Update positions
    positions %= L # Apply PBC
    new_forces, _ = calculate_forces(positions, N, L, r_cut)
    new_accelerations = new_forces / masses_col
    velocities += 0.5 * (accelerations + new_accelerations) * dt # Update velocities

    # --- Temperature Calculation ---
    kinetic_energy = 0.5 * np.sum(masses_col * velocities**2)
    current_T = (2.0 * kinetic_energy) / (3.0 * N)
    temperature_history.append(current_T)

    # --- Berendsen Thermostat ---
    if current_T > 1e-6 and not np.isnan(current_T):
         factor_inside_sqrt = 1.0 + (dt / tau) * (T / current_T - 1.0)
         if factor_inside_sqrt >= 0:
            scale_factor = np.sqrt(factor_inside_sqrt)
            velocities *= scale_factor # Scale velocities

    # --- Data Sampling ---
    if step % sample_interval == 0:
        big_particle_positions.append(positions[0].copy()) # Record big particle position

    # --- Progress Printout ---
    if step % 1000 == 0:
        print(f"Step {step}/{steps}, Temperature {current_T:.3f}")

print("Simulation done.")

# --- Plot Temperature ---
plt.figure()
plt.plot(np.arange(len(temperature_history)) * dt, temperature_history)
plt.axhline(T, color='r', linestyle='--', label='Target T')
plt.xlabel('Time (LJ units)')
plt.ylabel('Temperature (LJ units)')
plt.title('Temperature vs Time')
plt.legend()
plt.show()

# --- MSD Calculation & Plotting ---
big_particle_positions = np.array(big_particle_positions)

print("\nWARNING: MSD calculation uses simple unwrapping for PBC.")

# --- Unwrap Trajectory ---
unwrapped_positions = np.zeros_like(big_particle_positions)
unwrapped_positions[0] = big_particle_positions[0]
for i in range(1, len(big_particle_positions)):
    delta = big_particle_positions[i] - big_particle_positions[i-1]
    delta -= L * np.round(delta / L) # Minimum image displacement
    unwrapped_positions[i] = unwrapped_positions[i-1] + delta

# --- Calculate MSD ---
displacements_sq = np.sum((unwrapped_positions - unwrapped_positions[0])**2, axis=1)
times = np.arange(len(displacements_sq)) * dt * sample_interval

# --- Linear Regression ---
slope, intercept, r_value, p_value, std_err = linregress(times, displacements_sq)
D = slope / 6.0 # Calculate Diffusion Coefficient

# --- Plot MSD ---
plt.figure()
plt.plot(times, displacements_sq, 'bo', markersize=4, alpha=0.6, label='Simulation Data')
plt.plot(times, slope * times + intercept, 'r-', label=f'Linear Fit: $D={D:.4f}$')
plt.xlabel('Time (LJ units)')
plt.ylabel(r'Mean Squared Displacement $\langle r^2 \rangle$ (LJ units)')
plt.title('Brownian Motion - Mean Squared Displacement (MSD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print(f"\nCalculated Diffusion Coefficient (D): {D:.4f}")
print(f"Linear Fit R-squared: {r_value**2:.4f}")
