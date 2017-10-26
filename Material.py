import numpy as np

class Material:
    def __init__(self, youngs_modulus, poisson_ratio, thickness):
        self.E = youngs_modulus
        self.mu = poisson_ratio
        self.D = np.matrix([[1, self.mu, 0],
                            [self.mu, 1, 0],
                            [0, 0, (1 - self.mu) / 2]]) * (self.E / (1 - self.mu * self.mu))
        self.thickness = thickness
    