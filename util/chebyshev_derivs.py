from sympy import *

J, t = symbols('J t')
T = cos(J * acos(t)) 

for n in range(1, 5):
    dn_T_t = diff(T, t, n)
    print('d{}_T_t ='.format(n))
    print(dn_T_t)
    print()

