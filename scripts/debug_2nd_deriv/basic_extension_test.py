"""
Test the convergence rate of the extension.

usage: extension_test.py [-h] [-N N] [-c C] problem
"""
# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from numpy import sin, cos, sqrt
from scipy.special import jv
from scipy.integrate import quad
from scipy.optimize import brentq
import argparse

from scipy.optimize import brentq

import matplotlib.pyplot as plt

import sympy
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

def eval_curv(th):
    r = boundary.eval_r(th)
    x = r * np.cos(th)
    y = r * np.sin(th)
    return -1/(a**2 * b**2) * (x**2 / a**4 + y**2 / b**4)**(-3/2)

def get_arclength(th0, th1):
    get_t = lambda th: np.arctan(a/b*np.tan(th))

    eval_integrand = lambda t: np.sqrt(a**2*np.sin(t)**2 + b**2*np.cos(t)**2)
    t0 = get_t(th0)
    t1 = get_t(th1)

    return quad(eval_integrand, t0, t1)[0]

def eval_exp_d2_xi0_s(th0):
    ds = 0.001
    eval_exp = problem.eval_expected

    min1 = lambda th: get_arclength(th0, th) - ds
    th1 = brentq(min1, th0, th0 + ds*10)
    #print('arclength(th0-th1):', get_arclength(th0, th1))

    min2 = lambda th: get_arclength(th, th0) - ds
    th2 = brentq(min2, th0 - ds*10, th0+ds*10)
    #print('arclength(th2-th0):', get_arclength(th2, th0))

    r0 = boundary.eval_r(th0)
    x0 = r0*np.cos(th0)
    y0 = r0*np.sin(th0)

    r1 = boundary.eval_r(th1)
    x1 = r1*np.cos(th1)
    y1 = r1*np.sin(th1)

    r2 = boundary.eval_r(th2)
    x2 = r2*np.cos(th2)
    y2 = r2*np.sin(th2)

    return 1/ds**2 * (eval_exp(x1, y1) - 2*eval_exp(x0, y0) + eval_exp(x2, y2))

def run_test(h):
    th1 = np.pi/4
    r1 = boundary.eval_r(th1) + h

    n, th0 = boundary.get_boundary_coord(r1, th1)
    r0 = boundary.eval_r(th0)

    k = problem.k

    xi0 = problem.eval_bc(th0, 0)
    xi1 = problem.eval_d_u_outwards(th0, 0)

    d_th_s = boundary.eval_d_th_s(th0)
    d2_th_s = boundary.eval_d2_th_s(th0)

    d_xi0_th = float(eval_d_xi0_th(th0))
    d2_xi0_th = float(eval_d2_xi0_th(th0))

    d2_xi0_s = d2_xi0_th * d_th_s**2 + d_xi0_th * d2_th_s
    curv = eval_curv(th0)

    derivs = []
    derivs.append(xi0)
    derivs.append(xi1)

    '''print('-- Term 1 --')
    term1 = -k**2 * xi0
    print(term1)
    exp_term1 = -jv(0, r0)
    print('Error:', abs(term1 - exp_term1))

    print('\n-- Term 2 --')
    print(curv * xi1)

    print('\n-- Term 3 --')
    term3 = -d2_xi0_s
    print(term3)
    exp_term3 = -eval_exp_d2_xi0_s(th0)
    print('exp_term3:', exp_term3)
    print('Error:', abs(term3 - exp_term3))'''

    print()

    derivs.append(-k**2 * xi0 + curv * xi1 -d2_xi0_s)
    #print('2nd deriv=', derivs[-1])

    v = 0
    for l in range(len(derivs)):
        v += derivs[l] / math.factorial(l) * n**l

    exp = problem.eval_expected_polar(r1, th1)

    ###
    '''n = 0.0001
    eval_exp = problem.eval_expected
    normal = boundary.eval_normal(th0)*n

    x0 = r0*np.cos(th0)
    y0 = r0*np.sin(th0)

    x1 = x0 + normal[0]
    y1 = y0 + normal[1]

    x2 = x0 - normal[0]
    y2 = y0 - normal[1]

    d2 = 1/n**2 * (eval_exp(x1, y1) - 2*eval_exp(x0, y0) + eval_exp(x2, y2))
    print('2nd deriv should be=', d2)'''
    ###

    return abs(v - exp)

prec_str = '{:.5}'

parser = argparse.ArgumentParser()

io_util.add_arguments(parser, ('problem'))

args = parser.parse_args()
problem = problems.problem_dict[args.problem]()
problem.boundary = boundary = Ellipse(2.3)

s_th = sympy.symbols('th')
s_xi0 = sympy.besselj(0, boundary.r_expr).subs({'_a': a, '_b': b})

s_d_xi0_th = sympy.diff(s_xi0, s_th)
eval_d_xi0_th = sympy.lambdify(s_th, s_d_xi0_th)

s_d2_xi0_th = sympy.diff(s_xi0, s_th, 2)
eval_d2_xi0_th = sympy.lambdify(s_th, s_d2_xi0_th)

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
