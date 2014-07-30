from sympy import *
import numpy as np

from solver import cart_to_polar
from ps.ps import PizzaSolver

from .problem import Problem, Pizza

def get_L(R):
    return 2*R + 11/6*pi*R

def eval_bc_extended(R, r, th, sid): 
    L = get_L(R)

    if sid == 0:
        return R / L * (1 + th - pi/6)
    elif sid == 1:
        return 1 - r / L
    elif sid == 2:
        return r / L

class JumpNoCorrection(Pizza, Problem):
    k = 1

    solver_class = PizzaSolver
    homogeneous = True
    expected_known = False

    def eval_f(self, x, y):
        return 0
        
    def eval_bc(self, x, y):
        r, th = cart_to_polar(x, y)
        sid = Pizza.get_sid(th)
        return eval_bc_extended(self.R, r, th, sid)
        
    def eval_bc_extended(self, x, y, sid):
        r, th = cart_to_polar(x, y)
        return eval_bc_extended(self.R, r, th, sid)

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
    
def eval_phi(th):
    return th - pi/6
    
    
class RegF:

    def __init__(self):
        super().__init__()
        
        k, R, r, phi = symbols('k R r phi')
        args = (k, R, r, phi)
        f = get_reg_f_expr()
          
        self.f_polar_lambda = lambdify(args, f)
        
        d_f_r = diff(f, r)
        self.d_f_r_lambda = lambdify(args, d_f_r)
        
        d2_f_r = diff(f, r, 2)
        self.d2_f_r_lambda = lambdify(args, d2_f_r)
        
        d_f_th = diff(f, phi)
        self.d_f_th_lambda = lambdify(args, d_f_th)
        
        d2_f_th = diff(f, phi, 2)
        self.d2_f_th_lambda = lambdify(args, d2_f_th)
        
        d2_f_r_th = diff(f, r, phi)
        self.d2_f_r_th_lambda = lambdify(args, d2_f_r_th)
        
        x, y = symbols('x y')
        cart_args = (k, R, x, y)
        
        subs_dict_upper = {
            r: sqrt(x**2 + y**2),
            phi: atan2(y, x)
        }
        
        subs_dict_lower = {
            r: sqrt(x**2 + y**2),
            phi: atan2(y, x) + 2*pi
        }
        
        f_cart_upper = f.subs(subs_dict_upper)
        f_cart_lower = f.subs(subs_dict_lower)
        
        # Gradient
        d_f_x_upper = diff(f_cart_upper, x)
        self.d_f_x_upper_lambda = lambdify(cart_args, d_f_x_upper)
        
        d_f_x_lower = diff(f_cart_lower, x)
        self.d_f_x_lower_lambda = lambdify(cart_args, d_f_x_lower)
        
        d_f_y_upper = diff(f_cart_upper, y)
        self.d_f_y_upper_lambda = lambdify(cart_args, d_f_y_upper)
        
        d_f_y_lower = diff(f_cart_lower, y)
        self.d_f_y_lower_lambda = lambdify(cart_args, d_f_y_lower)
        
        # Hessian
        d2_f_x_upper = diff(f_cart_upper, x, 2)
        self.d2_f_x_upper_lambda = lambdify(cart_args, d2_f_x_upper)
        
        d2_f_x_lower = diff(f_cart_lower, x, 2)
        self.d2_f_x_lower_lambda = lambdify(cart_args, d2_f_x_lower)
        
        d2_f_x_y_upper = diff(f_cart_upper, x, y)
        self.d2_f_x_y_upper_lambda = lambdify(cart_args, d2_f_x_y_upper)
        
        d2_f_x_y_lower = diff(f_cart_lower, x, y)
        self.d2_f_x_y_lower_lambda = lambdify(cart_args, d2_f_x_y_lower)
        
        d2_f_y_upper = diff(f_cart_upper, y, 2)
        self.d2_f_y_upper_lambda = lambdify(cart_args, d2_f_y_upper)
        
        d2_f_y_lower = diff(f_cart_lower, y, 2)
        self.d2_f_y_lower_lambda = lambdify(cart_args, d2_f_y_lower)

    def eval_f_polar(self, r, th):
        assert th > .01
        return self.f_polar_lambda(self.k, self.R, r, eval_phi(th))
    
    def eval_d_f_r(self, r, th):
        assert th > .01
        return self.d_f_r_lambda(self.k, self.R, r, eval_phi(th))

    def eval_d2_f_r(self, r, th):
        assert th > .01
        return self.d2_f_r_lambda(self.k, self.R, r, eval_phi(th))

    def eval_d_f_th(self, r, th):
        assert th > .01
        return self.d_f_th_lambda(self.k, self.R, r, eval_phi(th))

    def eval_d2_f_th(self, r, th):
        assert th > .01
        return self.d2_f_th_lambda(self.k, self.R, r, eval_phi(th))

    def eval_d2_f_r_th(self, r, th):
        assert th > .01
        return self.d2_f_r_th_lambda(self.k, self.R, r, eval_phi(th))

    def eval_grad_f(self, x, y):
        if y > 0:
            d_f_x = self.d_f_x_upper_lambda(self.k, self.R, x, y)
            d_f_y = self.d_f_y_upper_lambda(self.k, self.R, x, y)
        else:
            d_f_x = self.d_f_x_lower_lambda(self.k, self.R, x, y)
            d_f_y = self.d_f_y_lower_lambda(self.k, self.R, x, y)
        
        return np.array((d_f_x, d_f_y))

    def eval_hessian_f(self, x, y):
        if y > 0:
            d2_f_x = self.d2_f_x_upper_lambda(self.k, self.R, x, y)
            d2_f_x_y = self.d2_f_x_y_upper_lambda(self.k, self.R, x, y)
            d2_f_y = self.d2_f_y_upper_lambda(self.k, self.R, x, y)
        else:
            d2_f_x = self.d2_f_x_lower_lambda(self.k, self.R, x, y)
            d2_f_x_y = self.d2_f_x_y_lower_lambda(self.k, self.R, x, y)
            d2_f_y = self.d2_f_y_lower_lambda(self.k, self.R, x, y)
        
        return np.array(((d2_f_x, d2_f_x_y), (d2_f_x_y, d2_f_y)))
          
    
class JumpReg(Pizza, RegF, Problem):
    k = 1

    expected_known = False
    
    def __init__(self):
        super().__init__()
        
        k, R, r, phi = symbols('k R r phi')     
        self.u04_lambda = lambdify((k, R, r, phi), get_u04_expr())
        
    def eval_bc(self, x, y):
        r, th = cart_to_polar(x, y)
        sid = Pizza.get_sid(th)
        return self.eval_bc_extended(x, y, sid)
        
    def eval_bc_extended(self, x, y, sid):
        a = self.a
        r, th = cart_to_polar(x, y)
        
        if sid != 1 and not 'from1':
            return 1
        
        if sid == 0 and th < a/2:
            th += 2*pi 
        elif sid == 1 and x < 0:    
            return self.eval_bc_extended(-x*cos(a), -x*sin(a), 2)
        elif sid == 2 and x < 0:
            return self.eval_bc_extended(r, 0, 1)
        
        bc_nc = eval_bc_extended(self.R, r, th, sid)
        u04 = self.u04_lambda(self.k, self.R, r, eval_phi(th))

        return float(bc_nc - u04)
        #return bc_nc
        #return u04 

