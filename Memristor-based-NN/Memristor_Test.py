import numpy as np
from Memristor import JoglekarMemristor
import matplotlib.pyplot as plt

memristor = JoglekarMemristor()

# PARAM
T = 1.0
dt = 1e-5

t = np.arange(0, T, dt)
v = np.sin(2 * np.pi * t)

resistance_values = []
current_values = []

# Simulation
for v_t in v:
    R = memristor.simulate(v_t, dt)
    I = v_t / R
    resistance_values.append(R)
    current_values.append(I)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(t, resistance_values)
plt.xlabel('Time (s)')
plt.ylabel('Resistance (Ohms)')
plt.title('Resistance vs Time')

plt.subplot(1, 2, 2)
plt.plot(v, current_values)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Current vs Voltage')

plt.tight_layout()
plt.show()
