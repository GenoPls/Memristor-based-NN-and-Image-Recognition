import numpy as np
class JoglekarMemristor:
    def __init__(self, D=10e-9, R_on=1e3, R_off=16e3, a=1, b=1, n=1):
        """"
        Parameters:
        D: Memristor width (in m)
        R_on: Resistance when the memristor is in the on state (in Ohms)
        R_off: Resistance when the memristor is in the off state (in Ohms)
        a, b: Parameters for the window function
        n: The exponent in the window function
        """
        self.D = D
        self.R_on = R_on
        self.R_off = R_off
        self.a = a
        self.b = b
        self.n = n
        # Initial memristor state
        self.w = 0.5 * D
        self.v_prev = 0

    def window(self, w):
        return self.a * (w / self.D)**self.n + self.b

    def resistance(self):
        R = self.R_on * (self.w / self.D) + self.R_off * (1 - self.w / self.D)
        return R

    def simulate(self, v, dt):
        """
        Parameters:
        v: The voltage across the memristor (in V)
        dt: The timestep (in s)
        """
        dw = (v - self.v_prev) / self.D * self.window(self.w) * dt
        self.w += dw
        self.v_prev = v

        # Make sure w stays within the memristor
        self.w = np.clip(self.w, 0, self.D)

        return self.resistance()
