import sys
import copy

from problems import RegularizeBc
from ps.ps import PizzaSolver, get_M, print_a_coef


default_primary_scheme_order = 4

class DualCoordinator:

    def __init__(self, problem, N, options):
        self.problem = copy.deepcopy(problem)
        self.N = N
        self.options = options

    def run_no_dual(self):
        solver = PizzaSolver(self.problem, self.N, self.options)
        return solver.run()

    def run_dual(self):
        scheme_order1 = self.options['scheme_order']
        scheme_order2 = scheme_order1 + 2

        if scheme_order2 == 6:
            print('Error: 6th order scheme has not be implemented. Exiting.')
            sys.exit(1)

        options2 = copy.copy(self.options)
        options2['scheme_order'] = scheme_order2
        options2['var_compute_a_only'] = True

        solver2 = PizzaSolver(self.problem, self.N, options2)
        a_coef = solver2.run()

        a_coef = a_coef[:get_M(scheme_order1)]
        print_a_coef(a_coef)
        print()

        self.problem.set_a_coef(a_coef)
        self.problem.regularize_bc = RegularizeBc.known

        options1 = copy.copy(self.options)
        options1['var_compute_a'] = False

        solver1 = PizzaSolver(self.problem, self.N, options1)
        return solver1.run()

    def run(self):
        if self.options['do_dual']:
            return self.run_dual()
        else:
            return self.run_no_dual()
