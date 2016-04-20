prec_str = '{:.5}'

import problems
import problems.boundary

def add_arguments(parser, args):
    if 'problem' in args:
        problem_choices = problems.problem_dict.keys()
        parser.add_argument('problem', metavar='problem',
            choices=problem_choices,
            help='name of the problem to run: ' + ', '.join(problem_choices))


    if 'boundary' in args:
        boundary_choices = problems.boundary.boundaries.keys()
        parser.add_argument('boundary', metavar='boundary',
            default='arc',
            choices=boundary_choices)

    if 'N' in args:
        parser.add_argument('-N', type=int, default=16,
            help='initial grid size')

    if 'c' in args:
        parser.add_argument('-c', type=int, nargs='?', const=128,
            help='run convergence test, up to the C x C grid. '
            'Default is 128.')

    if 'o' in args:
        parser.add_argument('-o', type=int,
            default=4, choices=(2, 4),
            help='order of scheme')

    if 'r' in args:
        parser.add_argument('-r', action='store_true',
            help='show relative convergence, even if the problem\'s '
            'true solution is known')


def print_options(options, N_list=None):
    print('[{} {} {}]'.format(options['problem'].name,
        options['boundary'].name,
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
