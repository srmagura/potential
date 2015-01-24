import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

k = 1

a = np.pi/6
nu = np.pi / (2*np.pi - a)

def eval_phi0(th):
    return -(th - np.pi/6) * (th - 2*np.pi) 

def calc_coef(M):
    coef = np.zeros(M)

    for m in range(1, M+1):
        if m % 2 == 0:
            coef[m-1] = 0
        else:
            coef[m-1] = 8/(2*np.pi - a) * 1/(m*nu)**3

    return coef

def do_test(M, prev_error):
    th_data = np.linspace(a, 2*np.pi, 1000)
    exact_data = np.array([eval_phi0(th) for th in th_data])
    
    expansion_data = np.zeros(len(th_data))
    coef = calc_coef(M) 

    for i in range(len(th_data)):
        th = th_data[i]

        phi0 = 0 
        for m in range(1, M+1):
            phi0 += coef[m-1] * np.sin(m*nu*(th - a))

        expansion_data[i] = phi0

    error = np.max(np.abs(expansion_data - exact_data))

    print('---- M = {} ----'.format(M))
    print('Fourier error:', error)

    if prev_error is not None:
        conv = np.log2(prev_error / error)
        print('Convergence:', conv)


    print()
    return error
    #plt.plot(th_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')
    #plt.plot(th_data, expansion_data, label='Expansion')

    #plt.show()


M = 4
prev_error = None

while M <= 128:
    prev_error = do_test(M, prev_error)
    M *= 2
