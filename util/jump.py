from sympy import *
import gen_problem_code as gpc

def get_u04_expr():
    k, R, r, phi = symbols('k R r phi')
    l = 1 / get_L(R)       
    k2 = k**2
    
    r, phi = symbols('r phi')
    r2 = r**2
    
    u04 = 44*k2*r2*pi*sqrt(3) * (k2*r2 - 12) * sin(2*phi)
    u04 += -11*pi*sqrt(3)*sin(4*phi) * k**4*r**4
    u04 += -396*pi*sin(phi)*l*r*(k2*r2 - 8)*sqrt(3)
    u04 += -1584*pi*sin(phi)**2 * cos(phi) * k2*l*r**3
    u04 += 1584*l*(k2*r2*cos(phi)**2 - 3/4*k2*r2 + 4)*pi*r*sin(phi)
    u04 += 3168*pi*cos(phi)*l*r
    u04 += 27*phi*(k2*r2 - 8)**2
    
    u04 *= 1 / (3168*pi)
    return u04
    
def get_reg_f_expr():
    k, R, r, phi = symbols('k R r phi')
    l = 1 / get_L(R)
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
    return f
    
    
    
gpc.find_f_polar_derivs(f)
    
x, y = symbols('x y')
f_cart = f.subs({
    r: sqrt(x**2 + y**2), 
    th: atan2(y, x)
})

gpc.find_f_total_derivs(f_cart)
