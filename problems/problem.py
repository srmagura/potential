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
        return self.eval_f_polar(*cart_to_polar(x, y))

    def eval_f_polar(self, r, th):
        x = r * cos(th)
        y = r * sin(th)
        return self.eval_f(x, y)

    def get_restore_polar(self, r, th):
        return 0
       
class Pizza:
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
        a = self.a
        
        if sid == 0:
            r = self.R
            th = arg
            
            if th < a/2:
                th += 2*np.pi
                 
        elif sid == 1:
            r = arg
            th = 2*np.pi
            
            if x < 0:    
                r = -x
                th = a
                sid = 2
                
        elif sid == 2:
            r = arg
            th = a

            if x < 0:
                th = 2*np.pi
                sid = 1

        return (r, th, sid)
        
    def eval_bc_extended(self, arg, sid):
        '''
        Extended versions of Dirichlet data for each segment. By default,
        the boundary data is taken as the trace of the true solution.
        Obviously this behaviour only makes sense if the true solution is
        known ahead of time.
        '''
        r, th, sid = self.wrap_func(arg, sid)
        return self.eval_expected_polar(r, th)

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
