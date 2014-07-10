from sympy import *

k, x, y = symbols('k x y')
r = sqrt(x**2 + y**2)
f = -k * sin(k*r)/r

hessian = (
    (diff(f, x, x), diff(f, x, y)),
    (diff(f, y, x), diff(f, y, y))
)

print(hessian)
