import datetime

from io_util import prec_str
from ps.ps import PizzaSolver

class Coordinator:

    def __init__(self, problem, N, options):
        self.problem = problem
        self.N = N

        self.options = options

    def print_info(self, N_list=None):
        print('[{} {}]'.format(self.problem.name, datetime.date.today()))

        options = self.options

        print('Scheme order: {}'.format(options['scheme_order']))

        print('k = ' + prec_str.format(float(self.problem.k)))
        print('R = ' + prec_str.format(self.problem.R))
        print('AD_len = ' + prec_str.format(self.problem.AD_len))
        print()

        if hasattr(self.problem, 'get_n_basis') and N_list:
            print('[Basis sets]')
            for N in N_list:
                print('{}: {}'.format(N, self.problem.get_n_basis(N=N)))
            print()

    def run(self):
        solver = PizzaSolver(self.problem, self.N, self.options)
        self.solver = solver
        return solver.run()

    def calc_rel_convergence(self, *args, **kwargs):
        return self.solver.calc_rel_convergence(*args, **kwargs)
