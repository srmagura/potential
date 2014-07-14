from sympy import *

def get_hessian(f):
    return (
        (diff(f, x, x), diff(f, x, y)),
        (diff(f, y, x), diff(f, y, y))
    )

k, x, y = symbols('k x y')
r = sqrt(x**2 + y**2)
f = -k * sin(k*r)/r
