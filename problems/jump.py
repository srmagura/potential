import numpy as np
from numpy import sin, cos, pi

from solver import cart_to_polar
from ps.ps import PizzaSolver

from .problem import Problem, Pizza

def get_L(R):
    return 2*R + 11/6*pi*R

def eval_bc_extended(R, r, th, sid): 
    L = get_L(R)

    if sid == 0:
        return R / L * (1 + th - pi/6)
    elif sid == 1:
        return (2*R + (2*pi - pi/6)*R - r) / L
    elif sid == 2:
        return r / L


class JumpNoCorrection(Pizza, Problem):
    k = 1

    solver_class = PizzaSolver
    homogeneous = True
    expectedKnown = False

    def eval_f(self, x, y):
        return 0
        
    def eval_bc(self, x, y):
        r, th = cart_to_polar(x, y)
        sid = Pizza.get_sid(th)
        return eval_bc_extended(self.R, r, th, sid)
        
    def eval_bc_extended(self, x, y, sid):
        r, th = cart_to_polar(x, y)
        return eval_bc_extended(self.R, r, th, sid)
