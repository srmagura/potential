import os
import subprocess
import itertools as it
from multiprocessing import Pool

C = 1024
OUT_DIR = '/Users/sam/Google\ Drive/research/output/'

problem_list = ['h-sine-range']#('h-hat', 'h-parabola', 'h-line-sine',)
boundary_list = ('arc', 'outer-sine', 'inner-sine', 'cubic', 'sine7',)

def worker(inputs):
    problem, boundary = inputs
    filename = OUT_DIR + problem + '/' + boundary + '.txt'

    command = 'python3 run.py {} {} -c {} '.format(problem, boundary, C)
    command += '> ' + filename
    subprocess.run(command, shell=True)
    print('Done: {} {}'.format(problem, boundary))

for problem_name in problem_list:
    _dir = OUT_DIR + problem_name
    os.makedirs(_dir, exist_ok=True)

with Pool(4) as p:
    p.map(worker, it.product(problem_list, boundary_list))
