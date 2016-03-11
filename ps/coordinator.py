import datetime

from io_util import prec_str
from ps.ps import PizzaSolver

class Coordinator:

    def __init__(self, problem, N, options={}):
        self.problem = problem
        self.N = N

        self.options = options

        solver = PizzaSolver(problem, N, options)
        self.solver = solver

    def print_info(self, N_list=None):
        options = self.options

        #TODO
        if 'boundary' in options:
            print('[{} {} {}]'.format(self.problem.name, options['boundary'].name,
                datetime.date.today()))
        else:
            print('[{} {}]'.format(self.problem.name,
                datetime.date.today()))

        if 'scheme_order' in options:
            print('Scheme order: {}'.format(options['scheme_order']))

        print('k = ' + prec_str.format(float(self.problem.k)))
        print('R = ' + prec_str.format(self.problem.R))
        print('a = ' + prec_str.format(self.problem.a))
        print('AD_len = ' + prec_str.format(self.problem.AD_len))
        print()

        if hasattr(self.problem, 'get_n_basis') and N_list:
            print('[Basis sets]')
            for N in N_list:
                print('{}: {}'.format(N, self.problem.get_n_basis(N)))
            print()

    def run(self):
        return self.solver.run()

    def calc_rel_convergence(self, *args, **kwargs):
        return self.solver.calc_rel_convergence(*args, **kwargs)
