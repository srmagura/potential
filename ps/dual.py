import sys
import copy
import datetime

from problems.singular import RegularizeBc
from problems.sing_h import SingH
from ps.ps import PizzaSolver, get_M, print_a_coef

prec_str = '{:.5}'

default_primary_scheme_order = 4

norm_names = ('l2', 'l2-wf1', 'sobolev')
default_norm = 'l2'

var_methods = ['fbterm', 'chebsine']
fft_test_var_methods = ('fft-test-chebsine', 'fft-test-fbterm')
var_methods.extend(fft_test_var_methods)

default_var_method = 'chebsine'


class DualCoordinator:

    def __init__(self, problem, N, options):
        self.problem = copy.deepcopy(problem)
        self.N = N

        if 'norm' not in options:
            options['norm'] = default_norm

        if 'var_method' not in options:
            options['var_method'] = default_var_method

        if 'var_compute_a' not in options:
            options['var_compute_a'] = False

        if 'print_a' not in options:
            options['print_a'] = False

        self.options = options

    def print_info(self, N_list=None):
        print('[{} {}]'.format(self.problem.name, datetime.date.today()))

        options = self.options
        print('var_compute_a = {}'.format(options['var_compute_a']))
        if options['var_compute_a']:
            print('Variational method:', options['var_method'])

        print('Norm:', options['norm'])

        def print_scheme(name, order):
            msg = '{}: {}'.format(name, order)
            msg += '  (M={})'.format(get_M(order))

            print(msg)

        if options['do_dual']:
            print_scheme('Secondary scheme order', options['scheme_order'] + 2)
            print_scheme('Primary scheme order', options['scheme_order'])
        else:
            print_scheme('Scheme order', options['scheme_order'])

        print('k = ' + prec_str.format(float(self.problem.k)))
        print('R = ' + prec_str.format(self.problem.R))
        print('AD_len = ' + prec_str.format(self.problem.AD_len))
        print()

        if hasattr(self.problem, 'get_n_basis') and N_list:
            print('[Basis sets]')
            for N in N_list:
                print('{}: {}'.format(N, self.problem.get_n_basis(N=N)))
            print()

    def run_no_dual(self):
        solver = PizzaSolver(self.problem, self.N, self.options)
        self.solver = solver
        return solver.run()

    def run_dual(self):
        scheme_order1 = self.options['scheme_order']
        scheme_order2 = scheme_order1 + 2

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

        a_coef = a_coef[:get_M(scheme_order1)]

        if self.options['print_a']:
            print_a_coef(a_coef)
            print()

        self.problem.set_a_coef(a_coef)
        self.problem.regularize_bc = RegularizeBc.known

        options1 = copy.copy(self.options)
        options1['var_compute_a'] = False
        #options1['M'] = x  # for 4-4

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
