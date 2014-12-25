import numpy as np
from numpy import cos, sin

from solver import cart_to_polar
from ps.ps import PizzaSolver

class Problem:
    homogeneous = False
    expected_known = True
    
    R = 2.3
    
    def __init__(self, **kwargs):
        super().__init__()

    def get_solver(self, *args, **kwargs):
        return self.solver_class(self, *args, **kwargs)
        
    def get_n_basis(self, N):
        if 'constant' in self.n_basis_dict:
            return self.n_basis_dict['constant']
        elif N in self.n_basis_dict:
            return self.n_basis_dict[N]
        else:
            return self.n_basis_dict[None]

    def eval_expected(self, x, y):
        r, th = cart_to_polar(x, y)
        
        if th < self.a/2:
            th += 2*np.pi
            
        return self.eval_expected_polar(r, th)

    def eval_f(self, x, y):
        if self.homogeneous:
            return 0
        
        return self.eval_f_polar(*cart_to_polar(x, y))

    def eval_f_polar(self, r, th):
        if self.homogeneous:
            return 0
        
        x = r * cos(th)
        y = r * sin(th)
        return self.eval_f(x, y)

    def get_restore_polar(self, r, th):
        return 0
    
       
class PizzaProblem(Problem):
    
    a = np.pi / 6
    solver_class = PizzaSolver
    
    # Semi-optimal values, determined by experiment
    n_basis_dict = {
        16: (21, 9), 
        32: (28, 8), 
        64: (34, 17), 
        128: (40, 24), 
        256: (45, 29),
        512: (53, 34)
    }
    
    shc_coef_len = 7
            
    def wrap_func(self, arg, sid):
        '''
        Used to create the smooth extensions of the Dirichlet data needed
        for the Chebyshev fit. Only valid when the Dirichlet data is smooth,
        even across the interfaces where the segments meet.
        '''
        if sid == 0:
            if arg < self.a/2:
                arg += 2*np.pi
                
        elif sid == 1 and arg < 0:
            arg = -arg
            sid = 2
            
        elif sid == 2 and arg < 0:
            arg = -arg
            sid = 1
                
        return (arg, sid)
        
    def arg_to_polar(self, arg, sid):
        if sid == 0:
            return (self.R, arg)
        elif sid == 1:
            return (arg, 2*np.pi)
        elif sid == 2:
            return (arg, self.a)

    def get_sid(self, th):
        '''
        Mapping of polar angles in the interval (0, 2*pi] to the set of 
        segment ID's, {0, 1, 2}.
        '''
        tol = 1e-12
        a = self.a

        if th >= 2*np.pi:
            return 1
        elif th > a:
            return 0
        elif abs(th - a) < tol:
            return 2

        assert False
