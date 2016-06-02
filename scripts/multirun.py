import os
import subprocess
import itertools as it
from multiprocessing import Pool

C = 1024

OUT_DIR_escape =  r'/Users/sam/Google\ Drive/research/output/'
OUT_DIR_noescape = r'/Users/sam/Google Drive/research/output/'

problem_list = ('trace-sine-range', 'shift-hat', 'shift-parabola',
    'shift-line-sine',)
#boundary_list = ('arc', 'outer-sine', 'inner-sine', 'cubic', 'sine7',)
boundary_list = ('cubic',)

def worker(inputs):
    problem, boundary = inputs
    filename = OUT_DIR_escape + problem + '/' + boundary + '.txt'

    command = 'python3 run.py {} {} -c {} '.format(problem, boundary, C)
    command += '> ' + filename
    #print(command)
    subprocess.run(command, shell=True)
    print('Done: {} {}'.format(problem, boundary))

for problem_name in problem_list:
    _dir = OUT_DIR_noescape + problem_name
    try:
        os.mkdir(_dir)
        print('Created directory: {}'.format(_dir))
    except FileExistsError:
        pass

with Pool(4) as p:
    p.map(worker, it.product(problem_list, boundary_list))
