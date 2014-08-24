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
from ps.multivalue import Multivalue


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
        
    def get_potential(self, ext):
        ww = ext.force_single_value_array()
        Lww = np.ravel(self.L.dot(ww))
        
        radius_Lw = {}
        
        for rsid in (1, 2):
            w = np.zeros((self.N-1)**2, dtype=complex)           
            gamma = self.all_gamma[rsid]
            
            for l in range(len(gamma)):
                i, j = gamma[l]
                index = matrices.get_index(self.N, i, j) 
                w[index] = ext.get(gamma[l], rsid)
                
            radius_Lw[rsid] = np.ravel(self.L.dot(w))
            
        rhs = np.zeros((self.N-1)**2, dtype=complex)
        
        for i, j in self.global_Mplus:
            index = matrices.get_index(self.N, i, j)
            x, y = self.get_coord(i, j)
            r, th = self.get_polar(i, j)
            
            # N must be at least 32. This may be invalid if a != pi/6
            if r < .6*self.R:
                in_Mplus1 = (i, j) in self.all_Mplus[1]
                in_Mplus2 = (i, j) in self.all_Mplus[2]
            
                if in_Mplus1 and not in_Mplus2:
                    rhs[index] = radius_Lw[1][index]
                    
                elif not in_Mplus1 and in_Mplus2:
                    rhs[index] = radius_Lw[2][index]
                    
                elif not in_Mplus1 and not in_Mplus2:
                    if y > 0:
                        sid = 2
                    else:
                        sid = 1
                        
                    rhs[index] = radius_Lw[sid][index]
                    
                else:
                    raise Exception('{}'.format((i, j)))
            else:
                rhs[index] = Lww[index]
                       
        return ext.add_array(-self.LU_factorization.solve(rhs))

    def get_Q(self, index):
        columns = []

        for JJ in range(len(self.B_desc)):
            ext = self.extend_basis(JJ, index)
            potential = self.get_potential(ext)
            projection = potential.get_gamma_array()

            columns.append(projection - ext.get_gamma_array())
        
        Q = np.column_stack(columns)
                    
        return Q
        
    def calc_c1(self):
        Q0 = self.get_Q(0)
        Q1 = self.get_Q(1)

        ext_f = self.extend_inhomo_f()           
        proj_f = self.get_potential(ext_f)
        
        term = proj_f.add_array(self.ap_sol_f).get_gamma_array()
        ext_f_array = ext_f.get_gamma_array()

        rhs = -Q0.dot(self.c0) - term + ext_f_array       
        self.c1 = np.linalg.lstsq(Q1, rhs)[0]

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
        self.ap_sol_f = self.LU_factorization.solve(self.B_src_f)
        #print('n_basis: {}'.format(n_basis_tuple))
 
        #return self.test_extend_src_f()
        #return self.test_extend_boundary()
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
        self.calc_c0()
        #self.calc_c1_exact()
        #self.c0_test()
        self.calc_c1()
        self.c1_test()
        #self.print_c1()
        #self.plot_gamma()
        #self.test_Q_system_residual()

        ext = self.extend_boundary()
        potential = self.get_potential(ext).add_array(self.ap_sol_f)
        u_act = potential.get_interior_array()

        #self.plot_contour(u_act)

        error = self.eval_error(u_act)
        
        result = Result()
        result.error = error
        result.u_act = u_act
        return result
