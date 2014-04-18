import numpy as np
import domain

class ComplexWave:
    # Wave number
    k = 1 
    kx = .8*k
    ky = .6*k

    def __init__(self):
        self.domain = domain.CircularDomain(self)

    # Boundary data
    def eval_bc(self, th):
        R = self.domain.R
        return self.eval_expected(R*np.cos(th), R*np.sin(th))

    # Expected solution
    def eval_expected(self, x, y):
        return np.exp(complex(0, self.kx*x + self.ky*y))

    # Expected normal derivative on boundary
    def eval_expected_derivative(self, th):
        x = complex(0, kx*np.cos(th) + ky*np.sin(th))
        return x*np.exp(R*x)
