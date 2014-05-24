from sympy import *

J, t = symbols('J t')
T = cos(J * acos(t)) 

d2_T_dt2 = diff(T, t, 2)
print('d2_T_dt2 =')
print(d2_T_dt2)
print()

d4_T_dt4 = diff(d2_T_dt2, t, 2)
print('d4_T_dt4 =')
print(d4_T_dt4)
