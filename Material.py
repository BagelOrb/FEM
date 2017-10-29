import numpy as np

class Material:
    def __init__(self, youngs_modulus, poisson_ratio, thickness):
        self.E = youngs_modulus
        self.mu = poisson_ratio
        self.D = np.matrix([[1      , self.mu, 0                 ], # strain_to_stress matrix
                            [self.mu, 1      , 0                 ],
                            [0      , 0      , (1 - self.mu) / 2]]) * (self.E / (1 - self.mu * self.mu))
        self.D_inv = np.matrix([[1       , -self.mu, 0], # stress_to_strain_matrix
                                [-self.mu, 1       , 0],
                                [0       , 0       , 2 + 2 * self.mu]]) * (1 / self.E)
        self.thickness = thickness
    
