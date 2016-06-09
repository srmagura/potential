import problems.functions as functions

from .singular import SingularProblem, HReg

from .sing_h_trace import _k, _n_basis_dict

class SingH_Shift(SingularProblem):

    homogeneous = True
    hreg = HReg.linsys

    k = _k

    n_basis_dict = _n_basis_dict

    def eval_bc(self, arg, sid):
        if sid == 0:
            return self.eval_phi0(arg)
        else:
            return 0


class Shift_Hat(SingH_Shift):

    m1_dict = {
        'arc': 128,
        'outer-sine': 134,
        'inner-sine': 220,
        'cubic': 210,
        'sine7': 200,
    }

    def eval_phi0(self, th):
        return functions.eval_hat_th(th)


class Shift_Parabola(SingH_Shift):

    m1_dict = {
        'arc': 48,
        'outer-sine': 200,
        'inner-sine': 220,
        'cubic': 140, #210,
        'sine7': 218,
    }

    def eval_phi0(self, th):
        return functions.eval_parabola(th, self.a)


class Shift_LineSine(SingH_Shift):

    m1_dict = {
        'arc': 8,
        'outer-sine': 210,
        'inner-sine': 236,
        'cubic': 200,
        'sine7': 204,
    }

    def eval_phi0(self, th):
        return functions.eval_linesine(th, self.a, self.nu)
