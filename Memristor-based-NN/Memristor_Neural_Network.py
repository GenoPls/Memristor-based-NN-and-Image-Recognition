import numpy as np
from Memristor import JoglekarMemristor

#One layered Memristor-based Neural Network
class MemristorNeuron:
    def __init__(self, num_inputs, threshold):
        self.memristors = [JoglekarMemristor() for _ in range(num_inputs)]
        self.threshold = threshold

    def forward(self, inputs, dt):
        currents = [memristor.simulate(v, dt) for memristor, v in zip(self.memristors, inputs)]
        total_current = np.sum(currents)
        output = 1 if total_current > self.threshold else 0

        return output
