import numpy as np
import matplotlib.pyplot as plt
import numba

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
r_cut_sq = r_cut**2 # Cutoff radius squared

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

# --- Initialize Velocities ---
velocities = np.random.randn(N, 3)
velocities -= np.mean(velocities, axis=0)
current_T = np.mean(np.sum(velocities**2, axis=1)) / 3
if current_T < 1e-10: current_T = T
velocities *= np.sqrt(T / current_T)
initial_T = np.mean(np.sum(velocities**2, axis=1)) / 3
print(f"Target T: {T}, Initial T set to: {initial_T:.6f}")

# --- Force Calculation (Numba JIT) ---
@numba.jit(nopython=True, fastmath=True, cache=True)
def calculate_forces(positions, N, L, r_cut_sq):
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    lower_bound_sq = 1e-12

    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r_vec -= np.round(r_vec / L) * L # PBC

            r_sq = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2 # Squared distance

            if r_sq < r_cut_sq and r_sq > lower_bound_sq: # Check distance
                norm_r = np.sqrt(r_sq)
                inv_r = 1.0 / norm_r
                inv_r6 = inv_r**6
                inv_r12 = inv_r6**2

                F_mag = 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r
                F = F_mag * r_vec # LJ Force

                forces[i] += F
                forces[j] -= F
                potential_energy += 4.0 * (inv_r12 - inv_r6) # LJ Potential

    return forces, potential_energy

# --- Data Storage ---
positions_history = []
velocities_history = []
potential_energy_history = []
kinetic_energy_history = []
v_mags = []
temperature_history = []

print("Starting simulation...")
# --- Main Simulation Loop ---
for step in range(steps):
    forces, potential_energy = calculate_forces(positions, N, L, r_cut_sq)
    potential_energy_history.append(potential_energy)

    accelerations = forces # m=1

    # --- Verlet Integrator ---
    positions_new = positions + velocities * dt + 0.5 * accelerations * dt**2
    positions_new %= L # Apply PBC
    positions = positions_new

    new_forces, _ = calculate_forces(positions, N, L, r_cut_sq)
    new_accelerations = new_forces

    velocities += 0.5 * (accelerations + new_accelerations) * dt # Update velocities

    # --- Temperature Calculation ---
    kinetic_energy = 0.5 * np.sum(velocities**2)
    kinetic_energy_history.append(kinetic_energy)
    current_T = (2.0 * kinetic_energy) / (3.0 * N)
    temperature_history.append(current_T)

    # --- Berendsen Thermostat ---
    if current_T > 1e-6 and not np.isnan(current_T):
         factor_inside_sqrt = 1.0 + (dt / tau) * (T / current_T - 1.0)
         if factor_inside_sqrt >= 0:
            scale_factor = np.sqrt(factor_inside_sqrt)
            velocities *= scale_factor # Scale velocities

    # --- Progress & Data Sampling ---
    if step % 1000 == 0:
        print(f"Step {step}/{steps}, Temperature {current_T:.3f}")

    if step > 2000:
        v_mags.extend(np.linalg.norm(velocities, axis=1))

    if step % 100 == 0:
        positions_history.append(positions.copy())
        velocities_history.append(velocities.copy())

print("done")

# --- Velocity Distribution Plot ---
v_mags = np.array(v_mags)
if len(v_mags) > 0:
    plt.hist(v_mags, bins=50, density=True, alpha=0.7, label='Simulation')
    v = np.linspace(0, max(v_mags), 100)
    avg_T = np.mean(temperature_history[2000:]) if len(temperature_history) > 2000 else T
    f_v = (np.sqrt(2 / np.pi) * v**2 / avg_T**(3/2)) * np.exp(-v**2 / (2 * avg_T)) # Maxwell-Boltzmann
    plt.plot(v, f_v, 'r-', label=f'Maxwell Distribution (T={avg_T:.2f})')
    plt.xlabel('Velocity (|v|)')
    plt.ylabel('Probability')
    plt.title('Velocity Distribution')
    plt.legend()
    plt.show()
else:
    print("No velocity data for histogram.")

# --- Temperature Plot ---
plt.figure()
plt.plot(temperature_history)
plt.axhline(T, color='r', linestyle='--', label='Target T')
plt.xlabel('Step')
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.legend()
plt.show()