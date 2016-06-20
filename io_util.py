import datetime

import problems
import problems.boundary

import ps.ode

prec_str = '{:.5}'

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


def print_options(options, meta_options={}):
    problem = options['problem']
    boundary = problem.boundary

    heading_items = [
        problem.name,
        boundary.name,
        str(datetime.date.today())
    ]

    if 'procedure_name' in meta_options:
        heading_items.insert(0, meta_options['procedure_name'])

    print('[{}]'.format(' '.join(heading_items)))

    if 'scheme_order' in options:
        print('Scheme order: {}'.format(options['scheme_order']))

    print('k = ' + prec_str.format(float(problem.k)))
    print('R = ' + prec_str.format(problem.R))
    print('a = ' + prec_str.format(problem.a))
    print('bet = ' + prec_str.format(boundary.bet))
    print('AD_len = ' + prec_str.format(problem.AD_len))
    if options.get('cheat_fft', False):
        print('!! cheat_fft = True !!')
    else:
        print('m1 = {}'.format(problem.get_m1()))
    print('fourier_N = {}'.format(ps.ode.fourier_N))
    print('ode_N = {}'.format(ps.ode.ode_N))
    print()

    if hasattr(problem, 'get_n_basis') and 'N_list' in meta_options:
        print('[Basis sets]')
        N_list = meta_options['N_list']
        for N in N_list:
            print('{}: {}'.format(N, problem.get_n_basis(N)))
        print()
