import matplotlib.pyplot as plt
from Memristor_Neural_Network import MemristorNeuron

neuron = MemristorNeuron(num_inputs=3, threshold=2)

# PARAM
inputs = [[0, 0, 0], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6], [0.5, 0.6, 0.7]]
dt = 1e-3
outputs = []

for input_comb in inputs:
    output = neuron.forward(input_comb, dt)
    outputs.append(output)

plt.figure(figsize=(10, 8))
plt.plot(range(len(inputs)), outputs, 'o')
plt.title('Output of Memristor-based Neuron')
plt.xlabel('Input Combination Index')
plt.ylabel('Output')
plt.xlim([-1, len(inputs)])
plt.ylim([-0.5, 1.5])
plt.grid(True)
plt.xticks(range(len(inputs)), labels=inputs)
plt.show()
