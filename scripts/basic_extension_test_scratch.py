from sympy import *

a, b, th = symbols('a b th')
r = a*b/(sqrt((b*cos(th))**2 + (a*sin(th))**2))

xi0 = sin(r*cos(th))
d2_xi0_th = diff(xi0, th, 2)
print('d2_xi0_th')
print(d2_xi0_th)

d_r_th = diff(r, th)
d_th_s = 1 / sqrt(r**2 + d_r_th**2)
print()
print('d_th_s')
print(d_th_s)
