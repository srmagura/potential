import numpy as np
from scipy.special import jv

from .problem import PizzaProblem

def eval_hat_x(x):
    """
    The so-called ``hat'' function that has something to do with
    distributions. Defined for x in the interval [-1, 1].
    """
    if abs(abs(x)-1) < 1e-15 or abs(x) > 1:
        return 0
    else:
        arg = -1/(1-x**2)
        return np.exp(arg)

def eval_hat_th(th):
    """
    The hat function scaled so that it can be used as boundary data on
    the arc. Defined for th in the interval [pi/6, 2*pi].
    """
    a = PizzaProblem.a
    x = (2*th-(2*np.pi+a))/(2*np.pi-a)
    return eval_hat_x(x)

def eval_parabola(th, a):
    return -(th - a) * (th - 2*np.pi)

def eval_linesine(th, k, R, a, nu):
    # FIXME ? (depends on k)

    return jv(nu/2, k*R)*((th - a)/(2*np.pi - a) - np.sin(nu/2*(th-a)))
