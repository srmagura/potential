import math
import numpy as np
from numpy import cos, sin

from solver import Solver, Result, cart_to_polar
import matrices

from ps.basis import PsBasis
from ps.grid import PsGrid
from ps.extend import PsExtend
from ps.inhomo import PsInhomo
from ps.debug import PsDebug


class PizzaSolver(Solver, PsBasis, PsGrid, PsExtend, PsInhomo, PsDebug):
    AD_len = 2*np.pi
    R = 2.3

    def __init__(self, problem, N, scheme_order, **kwargs):
        self.a = problem.a
        problem.R = self.R
        
        super().__init__(problem, N, scheme_order, **kwargs)
        
        self.ps_construct_grids()
           
    def get_sid(self, th):
        return self.problem.get_sid(th)

    def is_interior(self, i, j):
        r, th = self.get_polar(i, j)
        return r <= self.R and th >= self.a 

    def get_Q(self, index, ext_only=False):
        columns = []

        for JJ in range(len(self.B_desc)):
            ext = self.extend_basis(JJ, index)
            potential = self.get_potential(ext)
            projection = self.get_trace(potential)

            if not ext_only:
                columns.append(projection - ext)
            else:
                columns.append(ext)
        
        Q = np.column_stack(columns)
                    
        return Q

    def get_radius_point(self, sid, x, y):
        assert sid == 2

        m = np.tan(self.a)
        x1 = (x/m + y)/(m + 1/m)
        y1 = m*x1

        return (x1, y1) 

    def dist_to_radius(self, sid, x, y):
        assert sid == 2
        x1, y1 = self.get_radius_point(sid, x, y)
        return np.sqrt((x1-x)**2 + (y1-y)**2)

    def signed_dist_to_radius(self, sid, x, y):
        unsigned = self.dist_to_radius(sid, x, y)

        m = np.tan(self.a)
        if y > m*x:
            return -unsigned
        else:
            return unsigned
        
    def run(self):
        n_basis_tuple = self.problem.get_n_basis(self.N)
        #print('n_basis_tuple: {}'.format(n_basis_tuple))
        self.setup_B_desc(*n_basis_tuple)
        
        return self.test_extend_boundary()
        #return self.test_extend_boundary({
        #    (0, 'standard'),
        #    (0, 'left'),
        #    (0, 'right'),
        #    (1, 'standard'),
        #    (1, 'left'),
        #    (1, 'right'),
        #    (2, 'standard'),
        #    (2, 'left'),
        #    (2, 'right'),
        #})
        #return self.test_extend_basis()

        #self.calc_c0()
        #self.c0_test()
        #self.calc_c1()
        #self.c1_test()
        #self.print_c1()
        #self.test_extend_basis_not()
        return self.plot_gamma()
        #self.test_with_c1_exact()

        #ext = self.extend_boundary()
        #u_act = self.get_potential(ext) + self.ap_sol_f
        
        #self.plot_contour(u_act)

        #error = self.eval_error(u_act)
        
        result = Result()
        result.error = error
        result.u_act = u_act
        return result
