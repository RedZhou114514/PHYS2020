import numpy as np
import matplotlib.pyplot as plt
import numba

N = 100 
L = 10.0 
V = L ** 3
n = N / V
 
T = 1.0
dt = 0.001  
steps = 10000      
r_cut = 2.5
tau = 0.1

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
velocities = np.random.randn(N, 3)  
velocities -= np.mean(velocities, axis=0)
current_T = np.mean(np.sum(velocities**2, axis=1)) / 3
velocities *= np.sqrt(T / current_T)
initial_T = np.mean(np.sum(velocities**2, axis=1)) / 3
if np.abs(initial_T - T) < 1e-20:
    print("True")


@numba.jit(nopython=True)
def calculate_forces(positions, N, L, r_cut):
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j] 
            r_vec -= np.round(r_vec / L) * L
            norm_r = np.linalg.norm(r_vec) 
            if norm_r > 1e-5 and norm_r < r_cut:
                F = 24.0 * (2.0 * (1/norm_r)**13 - (1/norm_r)**7) * r_vec / norm_r
                forces[i] += F
                forces[j] -= F
                potential_energy += 4.0 * ((1/norm_r)**12 - (1/norm_r)**6)
    return forces, potential_energy

positions_history = [positions.copy()]
velocities_history = [velocities.copy()]
potential_energy_history = []
kinetic_energy_history = []
v_mags = []
temperature_history = []
for step in range(steps):
    forces, potential_energy = calculate_forces(positions, N, L, r_cut)
    potential_energy_history.append(potential_energy)

    accelerations = forces

    positions_new = positions + velocities * dt + 1/2 * accelerations * dt**2
    positions_new %= L  
    positions = positions_new

    new_forces, new_potential_energy = calculate_forces(positions, N, L, r_cut)
    new_accelerations = new_forces

    velocities += 0.5 * (accelerations + new_accelerations) * dt
    kinetic_energy = 0.5 * np.sum(np.sum(velocities**2, axis=1))
    kinetic_energy_history.append(kinetic_energy)
    current_T = (2.0 * kinetic_energy) / (3.0 * N)
    temperature_history.append(current_T)
    scale_factor = np.sqrt(1.0 + (dt / tau) * (T / current_T - 1.0))
    velocities *= scale_factor

    if step % 1000 == 0:
        print(f"步数 {step}, 温度 {current_T:.3f}")
    if step > 2000: 
        v_mags.extend(np.linalg.norm(velocities, axis=1))
    if step % 10 == 0:
        positions_history.append(positions.copy())
        velocities_history.append(velocities.copy())

    if step % 100 == 0:
        print(f"{step}/{steps}")
        
print("done")



v_mags = np.array(v_mags)
plt.hist(v_mags, bins=50, density=True, alpha=0.7, label='Simulation')
v = np.linspace(0, max(v_mags), 100)
f_v = (np.sqrt(2 / np.pi) * v**2 / T**(3/2)) * np.exp(-v**2 / (2 * T))
plt.plot(v, f_v, 'r-', label='Maxwell Distributin')
plt.xlabel('Velocity (|v|)')
plt.ylabel('Probability')
plt.legend()
plt.show()