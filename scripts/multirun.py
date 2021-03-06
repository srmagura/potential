"""
Script for running multiple problems on several different boundaries
with one command. Saves the output of each run to a file
"""
import os
import subprocess
import itertools as it
from multiprocessing import Pool

OUT_DIR_escape =  r'/Users/sam/Google\ Drive/research/output/'
OUT_DIR_noescape = r'/Users/sam/Google Drive/research/output/'

#problem_list = ('trace-sine-range', 'shift-hat', 'shift-parabola',
#    'shift-line-sine',)
problem_list = ('ihz-bessel-line',)
boundary_list = ('arc', 'outer-sine', 'inner-sine', 'cubic', 'sine7',)

args = '-c 1024'

def worker(inputs):
    problem, boundary = inputs
    filename = OUT_DIR_escape + problem + '/' + boundary + '.txt'

    command = 'python3 runz.py {} {} {}'.format(problem, boundary, args)
    command += ' > ' + filename
    #print(command)
    subprocess.call(command, shell=True)
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
