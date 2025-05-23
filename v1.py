import numpy as np
import matplotlib.pyplot as plt

N = 100          
L = 10.0 
V = L ** 3
n = N / V
 
T = 1.0   
dt = 0.005       
steps = 1000            
r_cut = 2.0

positions = np.random.rand(N, 3) * L
velocities = np.random.randn(N, 3)  
velocities -= np.mean(velocities, axis=0)
current_T = np.mean(np.sum(velocities**2, axis=1)) / 3
velocities *= np.sqrt(T / current_T)
initial_T = np.mean(np.sum(velocities**2, axis=1)) / 3
if np.abs(initial_T - T) < 1e-20:
    print("True")


def calculate_forces(positions):
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    for i in range(N):
        for j in range(i + 1, N): 
            r = positions[i] - positions[j]  
            r -= np.round(r / L) * L
            norm_r = np.linalg.norm(r)
            if norm_r < r_cut:
                F = 24.0 * (2.0 * (1/norm_r)**13 - (1/norm_r)**7) * r / norm_r
                forces[i] += F
                forces[j] -= F
                potential_energy += 4.0 * ((1/norm_r)**12 - (1/norm_r)**6)
    return forces, potential_energy

positions_history = [positions.copy()]
velocities_history = [velocities.copy()]
potential_energy_history = []
kinetic_energy_history = []
for step in range(steps):
    forces, potential_energy = calculate_forces(positions)
    potential_energy_history.append(potential_energy)

    accelerations = forces

    positions_new = positions + velocities * dt + 1/2 * accelerations * dt**2
    positions_new %= L  
    positions = positions_new

    new_forces, new_potential_energy = calculate_forces(positions)
    new_accelerations = new_forces

    velocities += 0.5 * (accelerations + new_accelerations) * dt
    kinetic_energy = 0.5 * np.sum(np.sum(velocities**2, axis=1))
    kinetic_energy_history.append(kinetic_energy)
    positions_history.append(positions.copy())
    velocities_history.append(velocities.copy())

    if step % 100 == 0:
        print(f"{step}/{steps}")
