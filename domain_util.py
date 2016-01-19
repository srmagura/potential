import numpy as np
import math

def cart_to_polar(x, y):
    """
    Convert Cartesian coordinates (x, y) to a polar coordinates (r, th).
    """
    r, th = math.hypot(x, y), math.atan2(y, x)

    if th <= 0:
        th += 2*np.pi

    return r, th

def wrap_func(a, arg, sid):
    """
    Can be used to create the smooth extensions of the Dirichlet data
    needed for the Chebyshev fit, for certain problems. In particular,
    the Dirichlet data needs to be sufficiently smooth at the reentrant
    corner.
    """
    if sid == 0 and arg < a/2:
            arg += 2*np.pi

    elif sid == 1 and arg < 0:
        arg = -arg
        sid = 2

    elif sid == 2 and arg < 0:
        arg = -arg
        sid = 1

    return (arg, sid)

def arg_to_polar(R, a, arg, sid):
    if sid == 0:
        return (R, arg)
    elif sid == 1:
        if arg >= 0:
            return (arg, 2*np.pi)
        else:
            return (-arg, np.pi)
    elif sid == 2:
        if arg >= 0:
            return (arg, a)
        else:
            return (-arg, np.pi + a)

def get_sid(a, th):
    """
    Mapping of polar angles in the interval (0, 2*pi] to the set of
    segment ID's, {0, 1, 2}.
    """
    tol = 1e-12

    if th >= 2*np.pi:
        return 1
    elif th > a:
        return 0
    elif abs(th - a) < tol:
        return 2
