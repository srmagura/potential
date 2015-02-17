import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy

a = np.pi/6
period = 2*np.pi - a
mult = 2*np.pi/period

def eval_phi0(th):
    x = (2*th-(2*np.pi+a))/(2*np.pi-a)

    if abs(abs(x)-1) < 1e-7 or abs(x) > 1:
        return 0
    else:
        arg = -1/(1-abs(x)**2)
        return np.exp(arg)

fourier_N = 1024
th_data = np.linspace(a, 2*np.pi, fourier_N)
exact_data = np.array([eval_phi0(th) for th in th_data])

expansion_data = np.zeros(len(th_data))

M = 15

coef_raw = np.fft.fft(exact_data)

J_dict = collections.OrderedDict(((J, None) for J in 
    range(-M, M+1)))

i = 0
coef = np.zeros(len(J_dict), dtype=complex)

for J in J_dict:
    J_dict[J] = i
    coef[i] = coef_raw[J] / fourier_N
    i += 1

for l in range(len(th_data)):
    th = th_data[l]

    phi0 = 0 
    for J, i in J_dict.items():
        exp = np.exp(complex(0, J*mult*(th-a)))
        phi0 += (coef[i] * exp).real

    expansion_data[l] = phi0

error = np.max(np.abs(expansion_data - exact_data))

print('---- #coef = {} ----'.format(2*M+1))
print('Fourier error:', error)

plt.plot(th_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')
plt.plot(th_data, expansion_data, label='Expansion')
#plt.plot(th_data, scipy.imag(expansion_data), label='Expansion imag')
plt.show()
