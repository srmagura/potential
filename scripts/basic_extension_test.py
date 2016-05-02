"""
Test the convergence rate of the extension.

usage: extension_test.py [-h] [-N N] [-c C] problem
"""
# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from numpy import sin, cos, sqrt
import argparse

from scipy.optimize import brentq

import matplotlib.pyplot as plt

import problems
from problems.boundary import Boundary
import io_util
import math

a = 2
b = 1

class Ellipse(Boundary):
    name = 'ellipse'
    r_expr_str = '_a*_b/(sqrt((_b*cos(th))**2 + (_a*sin(th))**2))'
    bet0 = 0.05
    additional_params = {'_a': a, '_b': b}

def eval_r(th):
    return a*b/(np.sqrt((b*np.cos(th))**2 + (a*np.sin(th))**2))

def eval_d_r_th(th):
    return a*b*(-a**2*sin(th)*cos(th) +
        b**2*sin(th)*cos(th))/(a**2*sin(th)**2 + b**2*cos(th)**2)**(3/2)

def eval_d2_xi0_th(th):
    return a*b*(-a*b*((a**2 - b**2)*cos(th)**2/(a**2*sin(th)**2 + b**2*cos(th)**2) + 1)**2*sin(th)**2*sin(a*b*cos(th)/sqrt(a**2*sin(th)**2 + b**2*cos(th)**2))/(a**2*sin(th)**2 + b**2*cos(th)**2) + (3*(a**2 - b**2)**2*sin(th)**2*cos(th)**2/(a**2*sin(th)**2 + b**2*cos(th)**2)**2 + 2*(a**2 - b**2)*sin(th)**2/(a**2*sin(th)**2 + b**2*cos(th)**2) - 1 + (a**2*sin(th)**2 - a**2*cos(th)**2 - b**2*sin(th)**2 + b**2*cos(th)**2)/(a**2*sin(th)**2 + b**2*cos(th)**2))*cos(th)*cos(a*b*cos(th)/sqrt(a**2*sin(th)**2 + b**2*cos(th)**2))/sqrt(a**2*sin(th)**2 + b**2*cos(th)**2))

def eval_d_th_s(th):
    return 1/sqrt(a**2*b**2/(a**2*sin(th)**2 + b**2*cos(th)**2) + a**2*b**2*(-a**2*sin(th)*cos(th) + b**2*sin(th)*cos(th))**2/(a**2*sin(th)**2 + b**2*cos(th)**2)**3)

def eval_curv(th):
    r = eval_r(th)
    x = r * np.cos(th)
    y = r * np.sin(th)
    return -1/(a**2 * b**2) * (x**2 / a**4 + y**2 / b**4)**(-3/2)

def get_boundary_coord(r1, th1):
    """
    Given a point with polar coordinates (r1, th1), find its
    coordinates (n, th) with respect to the boundary.
    """
    x1 = r1 * np.cos(th1)
    y1 = r1 * np.sin(th1)

    def eval_dist_deriv(th):
        # Let d = distance between (x0, y0) and (x1, y1)
        # This function returns the derivative of (1/2 d**2) wrt th
        r = eval_r(th)
        dr = eval_d_r_th(th)

        x0 = r * np.cos(th)
        y0 = r * np.sin(th)

        return ((x1 - x0) * (-dr*np.cos(th) + r*np.sin(th)) +
            (y1 - y0) * (-dr*np.sin(th) - r*np.cos(th)))

    # Optimization bounds
    diff = np.pi/6
    lbound = th1 - diff
    ubound = th1 + diff

    th0 = brentq(eval_dist_deriv, lbound, ubound, xtol=1e-16)

    # Get (absolute value of) n
    r = eval_r(th0)
    x0 = r * np.cos(th0)
    y0 = r * np.sin(th0)
    n = np.sqrt((x1-x0)**2 + (y1-y0)**2)

    return n, th0

def run_test(h):
    th1 = np.pi
    r1 = eval_r(th1) + h

    #n, th0 = get_boundary_coord(r1, th1)
    th0 = th1
    n = h

    k = problem.k

    xi0 = problem.eval_bc(th0, 0)
    xi1 = problem.eval_d_u_outwards(th0, 0)

    d_th_s = eval_d_th_s(th0)
    d2_xi0_s = d_th_s**2 * eval_d2_xi0_th(th0)

    curv = eval_curv(th0)

    derivs = []
    derivs.append(xi0)
    derivs.append(xi1)

    #print('2nd deriv=', -k**2 * xi0 + curv * xi1 - d2_xi0_s)
    #print('terms:')
    #print(-k**2 * xi0)
    #print(curv * xi1)
    #print(- d2_xi0_s)
    derivs.append(-k**2 * xi0 + curv * xi1 - d2_xi0_s)

    v = 0
    for l in range(len(derivs)):
        v += derivs[l] / math.factorial(l) * n**l

    exp = problem.eval_expected_polar(r1, th1)

    #h = 0.001
    #u = lambda t: problem.eval_expected_polar(r1, t)
    #d2 = 1/h**2 * (u(th0+h) - 2*u(th0) + u(th0-h))
    #print('2nd deriv should be=', d2)

    return abs(v - exp)

prec_str = '{:.5}'

parser = argparse.ArgumentParser()

io_util.add_arguments(parser, ('problem'))

args = parser.parse_args()
problem = problems.problem_dict[args.problem]()
problem.boundary = Ellipse(2.3)

h = 0.25
prev_error = None

while h > 0.01:
    print('---- h = {} ----'.format(h))

    error = run_test(h)
    print('Error: ' + prec_str.format(error))

    if prev_error is not None:
        convergence = np.log2(prev_error / error)
        print('Convergence: ' + prec_str.format(convergence))

    print()

    prev_error = error
    h /= 2
