from sympy import *
import gen_problem_code as gpc

phi, th, k, r, l = symbols('phi th k r l')
phi = th - pi / 6
k2 = k**2
r2 = r**2
    
f = k**4*r**3*pi*sqrt(3)*sin(2*phi)
f += -1/4*pi*sqrt(3)*sin(4*phi)*k**4*r**3
f += -9*pi*sin(phi)*l*r2*k2*sqrt(3)
f += 72*pi*sin(phi)**3*l
f += -36*pi*cos(phi)*l*(k2*r2 + 2)*sin(phi)**2 
f += 36*pi*l*((k2*r2 + 2)*cos(phi)**2 - 3/4*k2*r2 - 2)*sin(phi)
f += 27*k**4*phi*r**3/44
f += -72*pi*cos(phi)**3*l 
f += 72*pi*cos(phi)*l

f *= - k2*r/(72*pi)
    
gpc.find_f_polar_derivs(f)
    
x, y = symbols('x y')
f_cart = f.subs({
    r: sqrt(x**2 + y**2), 
    th: atan2(y, x)
})

gpc.find_f_total_derivs(f_cart)
