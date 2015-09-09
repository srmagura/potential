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

dir_name = 'regularity_test_7'

def run_test(m):
    print('m={} started.'.format(m))

    real_stdout = sys.stdout
    sys.stdout = open('{}/{}.txt'.format(dir_name, m), 'w')

    filler = '-'*20
    print('{0} m = {1} {0}'.format(filler, m))

    problem = SingH_Sine(m=m, M=0)
    interface = Interface(
        problem=problem,
        problem_name='sing-h-sine',
        #arg_string='-N 32 -c 1024',
        arg_string='-N 128'
    )
    interface.run()

    sys.stdout = real_stdout
    print('m={} done.'.format(m))

if os.path.exists(dir_name):
    print('Error: `{}` already exists'.format(dir_name))
    sys.exit(1)

os.mkdir(dir_name)

#with Pool(2) as pool:
#    pool.map(run_test, range(1, 9))
run_test(7)
