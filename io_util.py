prec_str = '{:.5}'

import problems

def add_arguments(parser, args):
    problem_choices = problems.problem_dict.keys()
    if 'problem' in args:
        parser.add_argument('problem', metavar='problem',
            choices=problem_choices,
            help='name of the problem to run: ' + ', '.join(problem_choices))

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
