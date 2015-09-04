"""
Determine how many terms of the Fourier-Bessel series must be removed
for the algorithm to regain convergence.
"""
# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import os
from multiprocessing import Pool

from interface import Interface
from problems.sing_h import SingH_Sine


def run_test(m):
    print('m={} started.'.format(m))

    real_stdout = sys.stdout
    sys.stdout = open('regularity_test/{}.txt'.format(m), 'w')

    filler = '-'*20
    print('{0} m = {1} {0}'.format(filler, m))

    problem = SingH_Sine(m=m, M=0)
    interface = Interface(
        problem=problem,
        problem_name='sing-h-sine',
        arg_string='-N 32 -c 1024',
    )
    interface.run()

    sys.stdout = real_stdout
    print('m={} done.'.format(m))

if os.path.exists('regularity_test'):
    print('Error: `regularity_test` already exists')
    sys.exit(1)

os.mkdir('regularity_test')

with Pool(4) as pool:
    pool.map(run_test, range(1, 9))
