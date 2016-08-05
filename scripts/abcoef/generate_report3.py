import json
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

problem = 'trace-line-sine'
boundary_list = ['inner-sine', 'cubic', 'outer-sine', 'sine7']
m1 = 200

table = []

def number_fmt(x):
    s = 'mony1{:.0e}mony2'.format(x)
    #s = s.replace('e', 'ehere')
    return s

for boundary in boundary_list:
    data = json.load(open('m1trend_{}_{}.dat'.format(problem, boundary)))

    i = data['m1_data'].index(m1)
    error = data['error'][i]
    new_error = [number_fmt(x) for x in error]
    table.append([boundary] + new_error)

headers = ['Boundary']
for m in range(1, 8):
    headers.append('acoef{}aend error'.format(m))

s = tabulate(table, headers=headers, tablefmt='latex_booktabs')
s = s.replace('acoef', '$a_').replace('aend', '$')
#s = s.replace('ehere', '\\text{e}')
s = s.replace('mony1', '\\num{')
s = s.replace('mony2', '}')
print(s)
