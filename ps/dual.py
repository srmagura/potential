import sys
import copy

from problems.singular import RegularizeBc
from problems.sing_h import SingH
from ps.ps import PizzaSolver, get_M, print_a_coef


default_primary_scheme_order = 4

class DualCoordinator:

    def __init__(self, problem, N, options):
        self.problem = copy.deepcopy(problem)
        self.N = N
        self.options = options

    def run_no_dual(self):
        solver = PizzaSolver(self.problem, self.N, self.options)
        self.solver = solver
        return solver.run()

    def run_dual(self):
        scheme_order1 = self.options['scheme_order']
        scheme_order2 = scheme_order1 #+ 2 #FIXME
        print('Using primary scheme order = 4')

        if scheme_order2 == 6:
            print('Error: 6th order scheme has not be implemented. Exiting.')
            sys.exit(1)

        problem = self.problem

        class Dual_SingH(SingH):

            k = problem.k
            n_basis_dict = problem.n_basis_dict

            def eval_phi0(self, th):
                return problem.to_dst(th)

        dual_sing_h = Dual_SingH()

        options2 = copy.copy(self.options)
        options2['scheme_order'] = scheme_order2
        options2['var_compute_a_only'] = True

        solver2 = PizzaSolver(dual_sing_h, self.N, options2)
        a_coef = solver2.run()

        #FIXME
        x=3#get_M(scheme_order1)
        a_coef = a_coef[:x]
        print_a_coef(a_coef)
        print()

        self.problem.set_a_coef(a_coef)
        self.problem.regularize_bc = RegularizeBc.known

        options1 = copy.copy(self.options)
        options1['var_compute_a'] = False
        options1['M'] = x  # can be deleted after 4-4 test is done

        solver1 = PizzaSolver(self.problem, self.N, options1)
        self.solver = solver1
        return solver1.run()

    def run(self):
        if self.options['do_dual']:
            return self.run_dual()
        else:
            return self.run_no_dual()

    def calc_rel_convergence(self, *args, **kwargs):
        return self.solver.calc_rel_convergence(*args, **kwargs)
